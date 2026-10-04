import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from components.block_halt_handler import block_halt_handler
from components.device_registry import device_identity_trust_registry
from components.event_collector import dma_event_collector
from components.iommu_enforcer import iommu_dma_remapping_enforcer
from components.memory_policy_engine import memory_map_policy_engine
from components.offline_learning import offline_learning_pipeline
from components.policy_rollout import policy_validation_rollout_service
from components.rate_limiter import dma_rate_limiter
from components.risk_model import dma_risk_model
from dma_guard import action_id, device_profile, memory_region


class TestComponents:
    def test_event_collector(self):
        collector = dma_event_collector()
        raw = {
            "device_id": "gpu-0",
            "address": "268435456",
            "size": "4096",
            "direction": "write",
            "privilege": "dma_buffer",
            "system_load": "0.35",
        }
        event = collector.collect(raw)
        assert event.device_id == "gpu-0"
        assert event.address == 268435456
        assert event.size == 4096
        assert event.direction == "write"
        assert event.privilege == "dma_buffer"
        assert pytest.approx(event.system_load, 0.001) == 0.35

    def test_device_registry(self):
        reg = device_identity_trust_registry()
        profile = device_profile("pci-1", "NVMe", True, ["storage"], ["read", "write"], 65536, 1000)
        assert reg.lookup("pci-1") is None
        reg.register(profile)
        assert reg.lookup("pci-1") == profile

    def test_block_halt_handler(self):
        handler = block_halt_handler()
        assert "dev-1" not in handler.revoked_devices

        handler.apply("dev-1", action_id.allow)
        assert "dev-1" not in handler.revoked_devices

        res = handler.apply("dev-1", action_id.block)
        assert res is True
        assert "dev-1" in handler.revoked_devices

        handler.apply("dev-2", action_id.throttle_halt)
        assert "dev-2" in handler.revoked_devices

    def test_iommu_enforcer(self):
        enforcer = iommu_dma_remapping_enforcer()
        assert enforcer.enforce("gpu-0", "gpu_framebuffer") == action_id.block
        enforcer.map_region("gpu-0", "gpu_framebuffer")
        assert enforcer.enforce("gpu-0", "gpu_framebuffer") == action_id.allow
        assert enforcer.enforce("gpu-0", "kernel_stack") == action_id.block

    def test_memory_policy_engine(self):
        reg = memory_region(0x1000, 0x5000, "dma_buffer", "test_reg")
        engine = memory_map_policy_engine([reg])
        resolved = engine.resolve(0x2000, 0x100)
        assert resolved == reg
        assert engine.resolve(0x6000, 0x100) is None

        profile = device_profile("dev", "GPU", True, ["test_reg"], ["write"], 4096, 100)
        assert engine.permitted(profile, reg, "write") is True
        assert engine.permitted(profile, reg, "read") is False

    def test_rate_limiter(self):
        limiter = dma_rate_limiter()
        now = 10_000_000_000
        for _ in range(5):
            assert limiter.exceeded("dev-x", 5, now_ns=now) is False
        # 6th request
        assert limiter.exceeded("dev-x", 5, now_ns=now) is True

    def test_offline_learning(self):
        model = dma_risk_model()
        pipeline = offline_learning_pipeline()

        # Dummy dataset: 8 samples of 6 features, labels in [0, 3]
        features = torch.randn(8, 6)
        labels = torch.randint(0, 4, (8,))
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss = pipeline.train_epoch(model, loader, optimizer)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_policy_rollout_service(self, sample_policy_file, tmp_path):
        service = policy_validation_rollout_service()
        validation = service.validate(str(sample_policy_file))
        assert validation["policy_valid"] is True
        assert validation["model_parameters"] < 20_000_000

        active_path = tmp_path / "active.json"
        active_path.write_text("{}", encoding="utf-8")
        h = service.rollout(str(sample_policy_file), str(active_path))
        assert len(h) == 64
