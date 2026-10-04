"""Synthetic DMA Telemetry Generator and Model Trainer.

Simulates realistic PCIe/Thunderbolt DMA transactions:
1. Normal workloads: GPU frame rendering, NIC packet transfers, NVMe block I/O (Label: 0 -> allow)
2. Throttling/Floods: Transfer bursts exceeding hardware limits, request storms (Label: 1 -> throttle_halt)
3. Direct Hardware Attacks: PCILeech-style CR3/page table scans, kernel memory access (Label: 2 -> block)
4. Subtle Anomalies: Suspicious unaligned buffers, anomalous load spikes, DMA race condition probes (Label: 3 -> log_alert)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from dma_guard import (
    action_id,
    dma_event,
    fast_feature_extractor,
    local_profile_store,
    policy_cache,
    small_risk_model,
)
from components.offline_learning import offline_learning_pipeline


def generate_synthetic_events(cache: policy_cache, count: int = 50_000, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

    # Approved regions from policy
    gpu_region = cache.store.regions.get("gpu_framebuffer")
    nic_region = cache.store.regions.get("nic_packet_buffer")
    kernel_region = cache.store.regions.get("kernel_stack")

    events_with_labels = []
    base_time = time.time_ns()

    for i in range(count):
        roll = random.random()
        sim_time = base_time + i * 1_000_000  # 1ms intervals

        # 1. Normal legitimate traffic (60%) -> Label 0 (allow)
        if roll < 0.60:
            if random.random() < 0.6:
                # GPU write/read to framebuffer
                offset = random.randint(0, (gpu_region.end - gpu_region.start - 65536) // 4096) * 4096
                address = gpu_region.start + offset
                size = random.choice([4096, 8192, 16384, 65536])
                dev_id = "gpu-0"
                direction = random.choice(["write", "read"])
            else:
                # NIC network packet write to ring buffer
                offset = random.randint(0, (nic_region.end - nic_region.start - 4096) // 2048) * 2048
                address = nic_region.start + offset
                size = random.choice([64, 128, 256, 512, 1500, 4096])
                dev_id = "nic-0"
                direction = "write"

            event = dma_event(
                device_id=dev_id,
                address=address,
                size=size,
                direction=direction,
                privilege="dma_buffer",
                timestamp_ns=sim_time,
                system_load=random.uniform(0.05, 0.60),
            )
            label = int(action_id.allow)

        # 2. Burst floods / Size violations (15%) -> Label 1 (throttle_halt)
        elif roll < 0.75:
            # Exceeds max transfer size or very high bursts under heavy load
            dev_id = random.choice(["gpu-0", "nic-0"])
            address = gpu_region.start if dev_id == "gpu-0" else nic_region.start
            size = random.choice([2_000_000, 16_000_000, 32_000_000])  # oversized
            event = dma_event(
                device_id=dev_id,
                address=address,
                size=size,
                direction="write",
                privilege="dma_buffer",
                timestamp_ns=sim_time,
                system_load=random.uniform(0.70, 0.99),
            )
            label = int(action_id.throttle_halt)

        # 3. Direct hardware attacks / PCILeech scans (15%) -> Label 2 (block)
        elif roll < 0.90:
            attack_sub_type = random.random()
            if attack_sub_type < 0.5:
                # Kernel stack / kernel memory probing
                address = kernel_region.start + random.randint(0, 4096)
                privilege = "kernel_stack"
            else:
                # Out-of-bounds unapproved physical addresses (memory dumping)
                address = random.randint(0x60000000, 0x80000000)
                privilege = random.choice(["kernel", "user"])

            event = dma_event(
                device_id=random.choice(["gpu-0", "nic-0", "pcileech-fpga", "thunderbolt-untrusted"]),
                address=address,
                size=random.choice([4096, 8192]),
                direction=random.choice(["read", "write"]),
                privilege=privilege,
                timestamp_ns=sim_time,
                system_load=random.uniform(0.20, 0.85),
            )
            label = int(action_id.block)

        # 4. Subtle anomalies / TOCTOU probes (10%) -> Label 3 (log_alert)
        else:
            # Inside approved buffer, but weirdly unaligned (e.g. byte offset 1337) and erratic load spikes
            unaligned_offset = random.randint(1, 4095)
            address = gpu_region.start + unaligned_offset
            event = dma_event(
                device_id="gpu-0",
                address=address,
                size=random.randint(128, 4096),
                direction="write",
                privilege="dma_buffer",
                timestamp_ns=sim_time,
                system_load=random.uniform(0.85, 0.99),
            )
            label = int(action_id.log_alert)

        events_with_labels.append((event, label))

    return events_with_labels


def build_and_save_dataset(
    policy_path: str = "policy.json",
    data_dir: str = "data",
    count: int = 50_000,
):
    path_data = Path(data_dir)
    path_data.mkdir(parents=True, exist_ok=True)

    store = local_profile_store.from_json(policy_path)
    cache = policy_cache(store)
    extractor = fast_feature_extractor(cache)

    print(f"Generating {count:,} synthetic DMA telemetry samples...")
    raw_events_and_labels = generate_synthetic_events(cache, count=count)

    feature_tensors = []
    label_tensors = []

    jsonl_sample_path = path_data / "sample_telemetry.jsonl"
    with jsonl_sample_path.open("w", encoding="utf-8") as sample_file:
        for idx, (event, label) in enumerate(raw_events_and_labels):
            feat = extractor.extract(event)
            feature_tensors.append(feat)
            label_tensors.append(label)

            # Write first 500 samples as readable JSONL for audit inspection
            if idx < 500:
                sample_file.write(
                    json.dumps({
                        "device_id": event.device_id,
                        "address": hex(event.address),
                        "size": event.size,
                        "direction": event.direction,
                        "privilege": event.privilege,
                        "system_load": round(event.system_load, 3),
                        "label": label,
                        "label_name": action_id(label).name,
                    })
                    + "\n"
                )

    X = torch.stack(feature_tensors)
    y = torch.tensor(label_tensors, dtype=torch.long)

    dataset_path = path_data / "dma_dataset.pt"
    torch.save({"features": X, "labels": y}, dataset_path)
    print(f"Saved dataset tensors to {dataset_path} (X: {X.shape}, y: {y.shape})")
    print(f"Saved readable JSONL samples to {jsonl_sample_path}")

    return X, y


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    checkpoint_out: str = "checkpoint.pt",
):
    dataset = TensorDataset(X, y)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = small_risk_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pipeline = offline_learning_pipeline()

    print(f"\nTraining small_risk_model ({sum(p.numel() for p in model.parameters()):,} parameters)...")
    for epoch in range(1, epochs + 1):
        loss = pipeline.train_epoch(model, train_loader, optimizer)

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for feats, targets in val_loader:
                preds = model(feats).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = (correct / total) * 100.0
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), checkpoint_out)
    print(f"Trained model checkpoint saved to {checkpoint_out}!")
    return model


if __name__ == "__main__":
    X, y = build_and_save_dataset(count=50_000)
    train_model(X, y, epochs=5, checkpoint_out="checkpoint.pt")
