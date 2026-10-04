import pytest
import torch
import torch.nn as nn
from dma_guard import (
    action_id,
    decision_controller,
    deterministic_rule_engine,
    dma_event,
    fast_feature_extractor,
)


class MockModel(nn.Module):
    def __init__(self, fixed_logits):
        super().__init__()
        self.fixed_logits = torch.tensor(fixed_logits, dtype=torch.float32)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.fixed_logits.unsqueeze(0).repeat(batch_size, 1)


class TestDecisionController:
    def test_short_circuit_on_rule_violation(self, test_cache):
        rule_engine = deterministic_rule_engine(test_cache)
        extractor = fast_feature_extractor(test_cache)

        # Mock model that would fail if called
        class FailingModel(nn.Module):
            def forward(self, x):
                raise AssertionError("Model forward should not be called when rule blocks")

        controller = decision_controller(rule_engine, extractor, FailingModel())
        # Violating event (kernel_stack)
        event = dma_event("gpu-0", 0xF0000000, 4096, "write", "kernel_stack")
        result = controller.decide(event)

        assert result.action == action_id.block
        assert result.risk == 1.0
        assert "protected kernel memory" in result.reason

    def test_decide_without_model(self, test_cache):
        rule_engine = deterministic_rule_engine(test_cache)
        extractor = fast_feature_extractor(test_cache)
        controller = decision_controller(rule_engine, extractor, model=None)

        event = dma_event("gpu-0", 0x10000000, 4096, "write", "dma_buffer")
        result = controller.decide(event)

        assert result.action == action_id.allow
        assert result.risk == 0.0

    def test_model_escalates_to_log_alert_when_threshold_exceeded(self, test_cache):
        rule_engine = deterministic_rule_engine(test_cache)
        extractor = fast_feature_extractor(test_cache)

        # Logits that strongly favor block (action 2)
        # softmax([0, 0, 10, 0]) -> prob[2] ~ 0.999
        mock_model = MockModel([0.0, 0.0, 10.0, 0.0])
        controller = decision_controller(
            rule_engine, extractor, mock_model, model_threshold=0.80
        )

        event = dma_event("gpu-0", 0x10000000, 4096, "write", "dma_buffer")
        result = controller.decide(event)

        assert result.action == action_id.log_alert
        assert "model anomaly recommendation: block" in result.reason
        assert result.risk > 0.80

    def test_model_does_not_escalate_when_below_threshold(self, test_cache):
        rule_engine = deterministic_rule_engine(test_cache)
        extractor = fast_feature_extractor(test_cache)

        # Logits give block around 50% (< 80% threshold)
        # softmax([1.0, 0.0, 1.2, 0.0])
        mock_model = MockModel([1.0, 0.0, 1.2, 0.0])
        controller = decision_controller(
            rule_engine, extractor, mock_model, model_threshold=0.80
        )

        event = dma_event("gpu-0", 0x10000000, 4096, "write", "dma_buffer")
        result = controller.decide(event)

        # Retains allow because confidence < 0.80
        assert result.action == action_id.allow
        assert "matches deterministic DMA policy" in result.reason
