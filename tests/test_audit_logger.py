import json
import pytest
from dma_guard import action_id, audit_logger, decision, dma_event


class TestAuditLogger:
    def test_log_creates_file_and_writes_jsonl(self, tmp_path):
        log_file = tmp_path / "nested" / "audit.jsonl"
        logger = audit_logger(log_file)

        event = dma_event(
            device_id="gpu-0",
            address=0x10000000,
            size=4096,
            direction="write",
            privilege="dma_buffer",
        )
        dec = decision(action_id.allow, "matches policy", 0.123, event)

        logger.log(dec)
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == int(action_id.allow)
        assert entry["reason"] == "matches policy"
        assert pytest.approx(entry["risk"], 0.001) == 0.123
        assert entry["event"]["device_id"] == "gpu-0"

    def test_log_appends_multiple_entries(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        logger = audit_logger(log_file)

        for i in range(3):
            event = dma_event(
                device_id=f"dev-{i}",
                address=0x10000000 + i * 0x1000,
                size=1024,
                direction="read",
                privilege="dma_buffer",
            )
            dec = decision(action_id.block, f"violation {i}", 1.0, event)
            logger.log(dec)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["event"]["device_id"] == f"dev-{i}"
            assert entry["action"] == int(action_id.block)
