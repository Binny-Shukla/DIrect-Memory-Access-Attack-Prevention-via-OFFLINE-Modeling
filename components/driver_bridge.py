"""Windows Kernel Driver & IOCTL Bridge for DMA Guard.

Provides:
1. WindowsKernelDriverBridge: Connects to Windows KMDF/WDM kernel driver (\\\\.\\DmaGuard)
   via Win32 CreateFileW and DeviceIoControl to intercept hardware DMA packets.
2. MockHardwareAttackBridge: A hardware testing harness simulating physical PCIe/FPGA
   attacks (PCILeech, ScreamerM2, CaptainDMA) to validate real attack scenarios.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import struct
import time
from typing import Optional

from dma_guard import action_id, dma_event


# Windows IOCTL code definitions
# CTL_CODE(FILE_DEVICE_UNKNOWN=0x22, Function, METHOD_BUFFERED=0, FILE_ANY_ACCESS=0)
def ctl_code(device_type: int, function: int, method: int = 0, access: int = 0) -> int:
    return (device_type << 16) | (access << 14) | (function << 2) | method


IOCTL_DMA_POLL_EVENT     = ctl_code(0x22, 0x800)  # Fetch pending DMA telemetry from driver
IOCTL_DMA_SEND_DECISION  = ctl_code(0x22, 0x801)  # Commit allow/block decision to IOMMU tables
IOCTL_DMA_REVOKE_DEVICE  = ctl_code(0x22, 0x802)  # Cut PCIe endpoint power / unmap domain
IOCTL_DMA_QUERY_STATUS   = ctl_code(0x22, 0x803)  # Query IOMMU VT-d / AMD-Vi status


class WindowsKernelDriverBridge:
    """Manages the Ring-3 to Ring-0 IOCTL channel to the Windows Kernel Driver."""

    DEVICE_NAME = r"\\.\DmaGuard"

    def __init__(self, device_path: str = DEVICE_NAME):
        self.device_path = device_path
        self.handle = None
        self.is_connected = False
        self._init_win32_api()

    def _init_win32_api(self):
        self.kernel32 = ctypes.windll.kernel32
        self.CreateFileW = self.kernel32.CreateFileW
        self.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
        ]
        self.CreateFileW.restype = wintypes.HANDLE

        self.DeviceIoControl = self.kernel32.DeviceIoControl
        self.DeviceIoControl.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
        ]
        self.DeviceIoControl.restype = wintypes.BOOL

        self.CloseHandle = self.kernel32.CloseHandle
        self.CloseHandle.argtypes = [wintypes.HANDLE]
        self.CloseHandle.restype = wintypes.BOOL

    def connect(self) -> bool:
        """Opens a communication handle to \\\\.\\DmaGuard."""
        GENERIC_READ_WRITE = 0xC0000000
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

        handle = self.CreateFileW(
            self.device_path,
            GENERIC_READ_WRITE,
            0,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None
        )
        if handle == INVALID_HANDLE_VALUE:
            self.is_connected = False
            return False

        self.handle = handle
        self.is_connected = True
        return True

    def send_verdict(self, device_id: str, action: action_id) -> bool:
        """Transmits the security verdict to Ring-0 to update the IOMMU translation table."""
        if not self.is_connected or not self.handle:
            return False

        # Payload: 64-byte device ID buffer + 4-byte integer action
        dev_bytes = device_id.encode("utf-8")[:63].ljust(64, b"\x00")
        payload = dev_bytes + struct.pack("I", int(action))
        bytes_returned = wintypes.DWORD(0)

        success = self.DeviceIoControl(
            self.handle,
            IOCTL_DMA_SEND_DECISION,
            payload,
            len(payload),
            None,
            0,
            ctypes.byref(bytes_returned),
            None
        )
        return bool(success)

    def close(self):
        if self.handle and self.is_connected:
            self.CloseHandle(self.handle)
            self.handle = None
            self.is_connected = False


class MockHardwareAttackBridge:
    """Hardware testing harness simulating raw PCIe TLP attack scenarios.

    Simulates actual hardware DMA attacks:
    1. ScreamerM2 / PCILeech linear physical RAM scanner (hunting for CR3 / page tables)
    2. Kernel code injection / patchguard bypass attempt
    3. Hotplug Thunderbolt rogue peripheral impersonation
    4. Legitimate high-throughput display stream
    """

    def __init__(self, enrolled_device_ids: list[str]):
        self.enrolled = set(enrolled_device_ids)

    def generate_attack_scenario(self, scenario_name: str) -> dma_event:
        now = time.time_ns()
        if scenario_name == "pcileech_page_scan":
            # PCILeech scans physical pages (e.g. 0x00001000 - 0x7FFFFFFF) to dump RAM
            return dma_event(
                device_id="PCI_FPGA_SCREAMER",
                address=0x70000000,
                size=4096,
                direction="read",
                privilege="user",
                timestamp_ns=now,
                system_load=0.15,
            )
        elif scenario_name == "kernel_code_injection":
            # Rogue PCIe device attempting to overwrite kernel code / IDT / CR3
            return dma_event(
                device_id="PCI_FPGA_SCREAMER",
                address=0xF0001000,
                size=512,
                direction="write",
                privilege="kernel_stack",
                timestamp_ns=now,
                system_load=0.40,
            )
        elif scenario_name == "thunderbolt_spoofing":
            # Untrusted hotplug device claiming to be an approved NIC
            return dma_event(
                device_id="PCI_ROUTER_ROGUE",
                address=0x40000000,
                size=16384,
                direction="write",
                privilege="dma_buffer",
                timestamp_ns=now,
                system_load=0.50,
            )
        elif scenario_name == "legitimate_gpu_stream":
            # Enrolled device accessing approved framebuffer
            dev = next((d for d in self.enrolled if "10DE" in d or "8086" in d), "PCI_8086_438B")
            return dma_event(
                device_id=dev,
                address=0x10000000,
                size=65536,
                direction="write",
                privilege="dma_buffer",
                timestamp_ns=now,
                system_load=0.25,
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
