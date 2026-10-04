import pytest
from knowledge import default_knowledge, knowledge_layer, knowledge_record, tokenize


class TestKnowledge:
    def test_tokenize(self):
        tokens = tokenize("Kernel Stack DMA Violation!")
        assert "kernel" in tokens
        assert "stack" in tokens
        assert "dma" in tokens
        assert "violation" in tokens

    def test_custom_knowledge_layer(self):
        records = [
            knowledge_record("rec-1", "GPU memory access buffer", ["gpu", "safe"]),
            knowledge_record("rec-2", "Kernel memory tampering attack", ["kernel", "attack"]),
        ]
        layer = knowledge_layer(records)

        results = layer.retrieve("kernel tampering")
        assert len(results) >= 1
        assert results[0].record_id == "rec-2"

    def test_default_knowledge_retrieval(self):
        results = default_knowledge.retrieve("thunderbolt hot-plug attack")
        assert len(results) >= 1
        assert any(r.record_id == "thunderbolt-dma" for r in results)

    def test_retrieve_no_match(self):
        results = default_knowledge.retrieve("completely unrelated zebra query xyz123")
        assert len(results) == 0
