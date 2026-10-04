"""Real Hardware Attack Simulation & Kernel Bridge Test.

Executes real-world hardware DMA attack vectors against the enrolled physical
PCIe policy baseline and the trained AI transformer risk model.
"""

from __future__ import annotations

from pathlib import Path
from dma_guard import build_guard
from components.hardware_scanner import generate_hardware_policy
from components.driver_bridge import MockHardwareAttackBridge, WindowsKernelDriverBridge


def run_hardware_attack_validation():
    policy_file = "policy_hardware.json"
    if not Path(policy_file).exists():
        print("[*] Generating enrolled physical hardware policy for this machine...")
        generate_hardware_policy(policy_file)

    checkpoint = "checkpoint.pt" if Path("checkpoint.pt").exists() else None
    print(f"\n[*] Initializing DMA Guard with {policy_file} and model: {checkpoint}...")
    guard, logger = build_guard(policy_file, "logs/hardware_audit.jsonl", checkpoint_path=checkpoint)

    # Attempt connection to real Windows kernel driver
    driver_bridge = WindowsKernelDriverBridge()
    print("[*] Probing Windows Kernel Driver (\\\\.\\DmaGuard)...")
    if driver_bridge.connect():
        print("[+] SUCCESS: Connected to live Windows Kernel Driver!")
    else:
        print("[-] Windows Kernel Driver (\\\\.\\DmaGuard) not installed in Ring-0.")
        print("[*] Activating MockHardwareAttackBridge for physical attack simulation...\n")

    # Load enrolled hardware device IDs
    import json
    policy_data = json.loads(Path(policy_file).read_text(encoding="utf-8"))
    enrolled_ids = [p["device_id"] for p in policy_data["device_profiles"]]

    attack_harness = MockHardwareAttackBridge(enrolled_ids)

    scenarios = [
        ("legitimate_gpu_stream", "Legitimate Host PCIe Peripheral Stream"),
        ("pcileech_page_scan", "PCILeech Physical Memory Scan (Dump RAM)"),
        ("kernel_code_injection", "Kernel Code / Stack Injection (PatchGuard Bypass)"),
        ("thunderbolt_spoofing", "Rogue Thunderbolt / Hot-plug PCIe Device Spoof"),
    ]

    print("=" * 80)
    print(f"{'TEST SCENARIO':<35} | {'VERDICT':<12} | {'RISK':<6} | {'REASON'}")
    print("=" * 80)

    for scenario_key, scenario_title in scenarios:
        event = attack_harness.generate_attack_scenario(scenario_key)
        decision = guard.decide(event)
        logger.log(decision)

        # Notify driver if connected
        if driver_bridge.is_connected:
            driver_bridge.send_verdict(event.device_id, decision.action)

        print(f"{scenario_title:<35} | {decision.action.name:<12} | {decision.risk:<6.3f} | {decision.reason}")

    print("=" * 80)
    print("\n[+] Audit decisions written to logs/hardware_audit.jsonl")


if __name__ == "__main__":
    run_hardware_attack_validation()
