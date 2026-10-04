"""Hardware PCI / PCIe / Thunderbolt Scanner for Windows.

Queries the operating system for real hardware devices connected over PCIe buses,
extracts Vendor ID (VEN), Device ID (DEV), and device class, and builds an enrolled
hardware baseline policy.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def scan_host_pci_devices():
    """Scans Windows PnP subsystem for active PCIe devices."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_PnPEntity | Where-Object { $_.DeviceID -like 'PCI*' } | "
        "Select-Object Name, DeviceID, Manufacturer, Service | ConvertTo-Json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_items = json.loads(proc.stdout)
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
    except Exception as exc:
        print(f"[-] Hardware scan warning: {exc}")
        raw_items = []

    devices = []
    for item in raw_items:
        dev_id_raw = item.get("DeviceID", "")
        match = re.search(r"VEN_([0-9A-Fa-f]{4})&DEV_([0-9A-Fa-f]{4})", dev_id_raw)
        if match:
            ven, dev = match.groups()
            clean_id = f"PCI_{ven.upper()}_{dev.upper()}"
        else:
            clean_id = dev_id_raw.replace("\\", "_").replace("&", "_")

        name = item.get("Name") or "Unknown Device"
        service = item.get("Service") or ""

        # Classify device type
        lower_name = name.lower()
        if any(k in lower_name for k in ["gpu", "display", "graphics", "nvidia", "radeon", "intel(r) uhd"]):
            dev_type = "GPU"
        elif any(k in lower_name for k in ["ethernet", "network", "nic", "wireless", "wi-fi", "gbe"]):
            dev_type = "NIC"
        elif any(k in lower_name for k in ["nvme", "nvm express", "storage", "rst", "sata"]):
            dev_type = "NVMe/Storage"
        elif any(k in lower_name for k in ["usb", "thunderbolt", "host controller", "xhci"]):
            dev_type = "Host_Controller"
        else:
            dev_type = "PCI_Peripheral"

        devices.append({
            "name": name,
            "device_id": clean_id,
            "raw_id": dev_id_raw,
            "type": dev_type,
            "service": service,
        })

    return devices


def generate_hardware_policy(out_path: str = "policy_hardware.json"):
    devices = scan_host_pci_devices()
    print(f"[*] Discovered {len(devices)} physical PCIe hardware devices on host.")

    # Base memory regions
    memory_regions = [
        {"start": 0x10000000, "end": 0x30000000, "privilege": "dma_buffer", "name": "gpu_framebuffer"},
        {"start": 0x40000000, "end": 0x48000000, "privilege": "dma_buffer", "name": "nic_packet_buffer"},
        {"start": 0x50000000, "end": 0x60000000, "privilege": "dma_buffer", "name": "nvme_dma_pool"},
        {"start": 0xF0000000, "end": 0xF0200000, "privilege": "kernel_stack", "name": "kernel_stack"},
    ]

    profiles = []
    for dev in devices:
        if dev["type"] == "GPU":
            allowed_regs = ["gpu_framebuffer"]
            max_size = 33554432  # 32MB
            max_rate = 50000
        elif dev["type"] == "NIC":
            allowed_regs = ["nic_packet_buffer"]
            max_size = 2097152   # 2MB
            max_rate = 100000
        elif dev["type"] == "NVMe/Storage":
            allowed_regs = ["nvme_dma_pool"]
            max_size = 4194304   # 4MB
            max_rate = 250000
        else:
            allowed_regs = ["gpu_framebuffer", "nic_packet_buffer"]
            max_size = 1048576   # 1MB
            max_rate = 10000

        profiles.append({
            "device_id": dev["device_id"],
            "device_type": dev["type"],
            "trusted": True,
            "allowed_regions": allowed_regs,
            "allowed_directions": ["read", "write"],
            "max_transfer_size": max_size,
            "max_requests_per_second": max_rate,
        })

    payload = {
        "memory_regions": memory_regions,
        "device_profiles": profiles,
    }

    out_file = Path(out_path)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[+] Enrolled hardware policy saved to: {out_file} ({len(profiles)} trusted profiles)")
    return payload


if __name__ == "__main__":
    generate_hardware_policy()
