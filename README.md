# DMA Guard

A zero-trust, policy-first Direct Memory Access (DMA) security monitor and attack mitigation system. DMA Guard combines deterministic physical memory access policies with an advisory lightweight Transformer neural network to detect, throttle, and block malicious PCIe and Thunderbolt peripheral DMA attacks (e.g., PCILeech, FPGA bus dumpers, and rogue hot-plug hardware).

---

## Architecture Overview

```text
[ PCIe / Thunderbolt Devices ]
              │
              ▼
    [ Hardware Telemetry ]
              │
              ▼
┌───────────────────────────────┐
│   Deterministic Rule Engine   │ ──(Violation: Kernel Stack, Out-of-bounds)──► [ INSTANT BLOCK ]
└─────────────┬─────────────────┘                                                (Latency: ~7.2 µs)
              │ (Allow Path)
              ▼
┌───────────────────────────────┐
│     Feature Extractor         │ (6-Dimensional Telemetry Tensor)
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ Small Transformer Risk Model  │ (100,804 parameters < 20M budget)
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│      Decision Controller      │ ──► [ ALLOW / LOG ALERT / THROTTLE ]
└───────────────────────────────┘       (Latency: ~1.06 ms)
```

1. **Deterministic Rule Engine (Fast-Path):** Non-negotiable security rules evaluated first. If an untrusted device accesses protected kernel memory or attempts an invalid transfer size, it is blocked immediately without requiring neural network inference.
2. **Advisory Risk Model (Transformer):** Analyzes legitimate-looking traffic for subtle anomalies (unaligned memory probes, unexpected load spikes, potential TOCTOU/race conditions).
3. **Hardware & Driver Bridge:** Enrolls physical host PCIe hardware IDs and integrates with Windows Kernel Mode (`\\.\DmaGuard`) via Win32 IOCTL.

---

## Offline Model Training: Epoch vs. Loss

The compact risk model (100,804 parameters) was trained on 50,000 synthetic PCIe transaction events using the AdamW optimizer and cross-entropy loss with gradient clipping:

```text
Loss
 │
0.038 ┼─ [Epoch 1: 0.0376]
      │   ●
0.030 ┼   │
      │   │
0.020 ┼   │
      │   │
0.010 ┼   │
      │   │
0.000 ┼───┴──────●──────────────●──────────────●──────────────●──────
      │      [Epoch 2:      [Epoch 3:      [Epoch 4:      [Epoch 5:
      │       0.0002]        0.0001]        0.0000]        0.0000]
      └─────────────────────────────────────────────────────────────►
        Epoch 1        Epoch 2        Epoch 3        Epoch 4        Epoch 5
```

### Numerical Training Progression

| Epoch | Cross-Entropy Loss |
| :---: | :---: |
| **1** | `0.0376` |
| **2** | `0.0002` |
| **3** | `0.0001` |
| **4** | `0.0000` |
| **5** | `0.0000` |

---

## Performance & Average Latency

Measured across 2,000 continuous hardware transaction decisions on an Intel host:

| Pipeline Stage | Average Latency (µs) | Average Latency (ms) | Description |
| :--- | :---: | :---: | :--- |
| **Fast-Path Rule Violation (Short-Circuit)** | **7.23 µs** | **0.0072 ms** | Instant policy block (kernel access, unapproved device) |
| **Full Pipeline (Rules + Extractor + Transformer)** | **1,061.27 µs** | **1.0613 ms** | Complete feature extraction + multi-layer transformer inference |

---

## Project Structure

```text
dma_guard/
├── dma_guard.py              # Core event collector, policy cache, rule engine, transformer model
├── demo.py                   # Verification script running trained checkpoint on test scenarios
├── generate_dataset.py       # 50,000-sample synthetic PCIe telemetry generator and trainer
├── run_hardware_attack_test.py # Hardware attack testing harness (PCILeech, kernel injection)
├── offline_pipeline.py       # Policy integrity checker and atomic rollout engine
├── knowledge.py              # Offline RAG-like attack knowledge lookup
├── policy.json               # Default local safe-memory and device profile store
├── policy_hardware.json      # Host-enrolled baseline generated from real system PCIe scan
├── components/
│   ├── hardware_scanner.py   # Windows PnP scanner discovering physical PCIe hardware devices
│   ├── driver_bridge.py      # Win32 IOCTL Ring-3 to Ring-0 interface (\\.\DmaGuard)
│   ├── event_collector.py    # Raw telemetry normalizer
│   ├── device_registry.py    # Trusted device identity and capability store
│   ├── memory_policy_engine.py# Address range resolver and permission engine
│   ├── rule_engine.py        # Hard policy evaluation
│   ├── feature_extractor.py  # 6-dim tensor feature transformation
│   ├── risk_model.py         # 4-action transformer classifier
│   ├── decision_controller.py# Security verdict coordinator
│   ├── rate_limiter.py       # Sliding 1-second transfer rate limiter
│   ├── block_halt_handler.py # Revoked session store
│   ├── audit_logger.py       # Append-only JSONL decision logger
│   └── offline_learning.py   # Supervised training loop
└── tests/                    # 42 unit and integration tests (100% passing)
```

---

## Quickstart & Usage

### 1. Run Verification Demo
```powershell
python demo.py
```

### 2. Generate Synthetic Dataset & Train Transformer
```powershell
python generate_dataset.py
```

### 3. Run Physical Hardware Attack Test
```powershell
python run_hardware_attack_test.py
```

### 4. Run the Pytest Suite
```powershell
python -m pytest -v
```

---

## License

This project is licensed under the terms of the **GNU General Public License v3.0** (GNU GPLv3). See the [LICENSE](LICENSE) file for complete details.
