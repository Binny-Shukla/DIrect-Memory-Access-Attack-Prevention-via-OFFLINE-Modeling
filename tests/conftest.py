import json
import pytest
from pathlib import Path
import sys

# Ensure dma_guard directory and components are on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dma_guard import (
    action_id,
    build_guard,
    device_profile,
    dma_event,
    local_profile_store,
    memory_region,
    policy_cache,
    deterministic_rule_engine,
    fast_feature_extractor,
    small_risk_model,
    decision_controller,
)


@pytest.fixture
def sample_policy_dict():
    return {
        "memory_regions": [
            {
                "start": 0x10000000,
                "end": 0x11000000,
                "privilege": "dma_buffer",
                "name": "gpu_framebuffer",
            },
            {
                "start": 0x20000000,
                "end": 0x20400000,
                "privilege": "dma_buffer",
                "name": "nic_packet_buffer",
            },
            {
                "start": 0xF0000000,
                "end": 0xF0100000,
                "privilege": "kernel_stack",
                "name": "kernel_stack",
            },
        ],
        "device_profiles": [
            {
                "device_id": "gpu-0",
                "device_type": "GPU",
                "trusted": True,
                "allowed_regions": ["gpu_framebuffer"],
                "allowed_directions": ["read", "write"],
                "max_transfer_size": 8388608,
                "max_requests_per_second": 30000,
            },
            {
                "device_id": "nic-0",
                "device_type": "NIC",
                "trusted": True,
                "allowed_regions": ["nic_packet_buffer"],
                "allowed_directions": ["read", "write"],
                "max_transfer_size": 1048576,
                "max_requests_per_second": 100,
            },
            {
                "device_id": "untrusted-dev",
                "device_type": "USB",
                "trusted": False,
                "allowed_regions": ["gpu_framebuffer"],
                "allowed_directions": ["read"],
                "max_transfer_size": 4096,
                "max_requests_per_second": 10,
            },
        ],
    }


@pytest.fixture
def sample_policy_file(tmp_path, sample_policy_dict):
    policy_file = tmp_path / "test_policy.json"
    policy_file.write_text(json.dumps(sample_policy_dict), encoding="utf-8")
    return policy_file


@pytest.fixture
def profile_store(sample_policy_dict):
    profiles = [device_profile(**item) for item in sample_policy_dict["device_profiles"]]
    regions = [memory_region(**item) for item in sample_policy_dict["memory_regions"]]
    return local_profile_store(profiles, regions)


@pytest.fixture
def test_cache(profile_store):
    return policy_cache(profile_store)


@pytest.fixture
def rule_engine(test_cache):
    return deterministic_rule_engine(test_cache)
