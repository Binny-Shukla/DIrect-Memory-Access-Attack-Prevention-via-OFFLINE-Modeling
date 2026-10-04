import pytest
from dma_guard import action_id, dma_event, memory_region


class TestMemoryRegion:
    def test_region_contains_within_bounds(self):
        region = memory_region(start=0x1000, end=0x2000, privilege="dma_buffer", name="test_region")
        assert region.contains(0x1000, 0x1000) is True
        assert region.contains(0x1500, 0x100) is True
        assert region.contains(0x1000, 1) is True

    def test_region_contains_out_of_bounds(self):
        region = memory_region(start=0x1000, end=0x2000, privilege="dma_buffer", name="test_region")
        # Starts before
        assert region.contains(0x0FFF, 0x100) is False
        # Starts after
        assert region.contains(0x2000, 0x10) is False
        # Overflows end boundary
        assert region.contains(0x1F00, 0x200) is False


class TestDeterministicRuleEngine:
    def test_allow_valid_request(self, rule_engine):
        event = dma_event(
            device_id="gpu-0",
            address=0x10000000,
            size=4096,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.allow
        assert "matches deterministic DMA policy" in reason

    def test_block_unknown_device(self, rule_engine):
        event = dma_event(
            device_id="unknown-device-999",
            address=0x10000000,
            size=4096,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "unknown or untrusted device" in reason

    def test_block_untrusted_device(self, rule_engine):
        event = dma_event(
            device_id="untrusted-dev",
            address=0x10000000,
            size=4096,
            direction="read",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "unknown or untrusted device" in reason

    def test_block_invalid_transfer_size(self, rule_engine):
        for invalid_size in [0, -1, -4096]:
            event = dma_event(
                device_id="gpu-0",
                address=0x10000000,
                size=invalid_size,
                direction="write",
                privilege="dma_buffer",
            )
            action, reason = rule_engine.evaluate(event)
            assert action == action_id.block
            assert "invalid transfer size" in reason

    def test_block_address_outside_approved_regions(self, rule_engine):
        # 0x30000000 is not in any memory region
        event = dma_event(
            device_id="gpu-0",
            address=0x30000000,
            size=4096,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "outside every approved DMA region" in reason

    def test_block_protected_kernel_memory(self, rule_engine):
        # 0xF0000000 is kernel_stack
        event = dma_event(
            device_id="gpu-0",
            address=0xF0000000,
            size=4096,
            direction="write",
            privilege="kernel_stack",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "protected kernel memory is never DMA accessible" in reason

    def test_block_unapproved_region_for_device(self, rule_engine):
        # gpu-0 accessing nic_packet_buffer (0x20000000)
        event = dma_event(
            device_id="gpu-0",
            address=0x20000000,
            size=4096,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "device is not approved for this memory region" in reason

    def test_block_unapproved_direction(self, rule_engine, profile_store, test_cache):
        # Modify nic-0 to read-only
        profile_store.profiles["nic-0"].allowed_directions = ["read"]
        event = dma_event(
            device_id="nic-0",
            address=0x20000000,
            size=1024,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.block
        assert "DMA direction is not approved for device" in reason

    def test_throttle_size_limit_exceeded(self, rule_engine):
        # nic-0 max_transfer_size is 1,048,576
        event = dma_event(
            device_id="nic-0",
            address=0x20000000,
            size=2_000_000,
            direction="write",
            privilege="dma_buffer",
        )
        action, reason = rule_engine.evaluate(event)
        assert action == action_id.throttle_halt
        assert "transfer exceeds device size limit" in reason

    def test_throttle_rate_limit_exceeded(self, rule_engine):
        # nic-0 limit is 100 requests per second
        base_time = 1_000_000_000
        for i in range(100):
            event = dma_event(
                device_id="nic-0",
                address=0x20000000,
                size=1024,
                direction="read",
                privilege="dma_buffer",
                timestamp_ns=base_time + i * 1_000,
            )
            action, _ = rule_engine.evaluate(event)
            assert action == action_id.allow

        # 101st request in the same second window
        exceeded_event = dma_event(
            device_id="nic-0",
            address=0x20000000,
            size=1024,
            direction="read",
            privilege="dma_buffer",
            timestamp_ns=base_time + 101 * 1_000,
        )
        action, reason = rule_engine.evaluate(exceeded_event)
        assert action == action_id.throttle_halt
        assert "device exceeds request-rate limit" in reason
