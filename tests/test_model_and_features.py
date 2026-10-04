import math
import pytest
import torch
from dma_guard import dma_event, fast_feature_extractor, small_risk_model


class TestFastFeatureExtractor:
    def test_feature_extraction_allowed_event(self, test_cache):
        extractor = fast_feature_extractor(test_cache)
        event = dma_event(
            device_id="gpu-0",
            address=0x10000800,  # 2048 offset in 4k page
            size=4096,
            direction="write",
            privilege="dma_buffer",
            system_load=0.45,
        )
        features = extractor.extract(event)

        assert features.shape == (6,)
        assert features.dtype == torch.float32

        expected_size_feature = math.log2(4096) / 32.0  # 12 / 32 = 0.375
        assert pytest.approx(features[0].item(), 0.001) == expected_size_feature
        assert pytest.approx(features[1].item(), 0.001) == 0.45
        assert features[2].item() == 1.0  # write
        assert features[3].item() == 0.0  # approved region found
        assert features[4].item() == 0.0  # not kernel
        assert pytest.approx(features[5].item(), 0.001) == 2048 / 4096.0

    def test_feature_extraction_unapproved_region_and_kernel(self, test_cache):
        extractor = fast_feature_extractor(test_cache)
        # Outside any region, read direction, kernel privilege
        event = dma_event(
            device_id="gpu-0",
            address=0x50000000,
            size=1024,
            direction="read",
            privilege="kernel",
            system_load=0.80,
        )
        features = extractor.extract(event)

        assert features[2].item() == 0.0  # read
        assert features[3].item() == 1.0  # region is None
        assert features[4].item() == 1.0  # kernel privilege


class TestSmallRiskModel:
    def test_parameter_count_within_budget(self):
        model = small_risk_model()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert param_count < 20_000_000

    def test_forward_pass_2d(self):
        model = small_risk_model()
        model.eval()
        dummy_input = torch.randn(4, 6)
        with torch.inference_mode():
            out = model(dummy_input)
        assert out.shape == (4, 4)

    def test_forward_pass_3d(self):
        model = small_risk_model()
        model.eval()
        dummy_input = torch.randn(4, 8, 6)
        with torch.inference_mode():
            out = model(dummy_input)
        assert out.shape == (4, 4)
