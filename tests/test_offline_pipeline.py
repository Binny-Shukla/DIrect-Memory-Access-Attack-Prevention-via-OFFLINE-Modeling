import hashlib
import json
from pathlib import Path
import pytest
from offline_pipeline import rollout_policy, validate_model, validate_policy


class TestOfflinePipeline:
    def test_validate_default_project_policy(self):
        policy_path = Path(__file__).resolve().parent.parent / "policy.json"
        assert validate_policy(policy_path) is True

    def test_validate_policy_missing_regions_or_profiles(self, tmp_path):
        bad_policy = tmp_path / "bad.json"
        bad_policy.write_text(json.dumps({"memory_regions": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="policy requires memory regions and device profiles"):
            validate_policy(bad_policy)

    def test_validate_policy_invalid_range(self, tmp_path):
        bad_policy = tmp_path / "bad_range.json"
        bad_policy.write_text(
            json.dumps({
                "memory_regions": [{"start": 100, "end": 50, "privilege": "dma", "name": "reg"}],
                "device_profiles": [{"device_id": "d1", "allowed_regions": ["reg"], "max_transfer_size": 10, "max_requests_per_second": 10}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid range"):
            validate_policy(bad_policy)

    def test_validate_policy_unknown_region_in_profile(self, tmp_path):
        bad_policy = tmp_path / "bad_unknown.json"
        bad_policy.write_text(
            json.dumps({
                "memory_regions": [{"start": 10, "end": 50, "privilege": "dma", "name": "reg"}],
                "device_profiles": [{"device_id": "d1", "allowed_regions": ["non_existent_reg"], "max_transfer_size": 10, "max_requests_per_second": 10}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown region in profile"):
            validate_policy(bad_policy)

    def test_validate_policy_invalid_limits(self, tmp_path):
        bad_policy = tmp_path / "bad_limits.json"
        bad_policy.write_text(
            json.dumps({
                "memory_regions": [{"start": 10, "end": 50, "privilege": "dma", "name": "reg"}],
                "device_profiles": [{"device_id": "d1", "allowed_regions": ["reg"], "max_transfer_size": -1, "max_requests_per_second": 10}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid limits"):
            validate_policy(bad_policy)

    def test_validate_model_parameters(self):
        param_count = validate_model()
        assert param_count > 0
        assert param_count < 20_000_000

    def test_rollout_policy_atomic(self, tmp_path, sample_policy_file):
        active_file = tmp_path / "active_policy.json"
        active_file.write_text("{}", encoding="utf-8")

        result_hash = rollout_policy(sample_policy_file, active_file)
        expected_hash = hashlib.sha256(sample_policy_file.read_bytes()).hexdigest()

        assert result_hash == expected_hash
        assert active_file.read_text(encoding="utf-8") == sample_policy_file.read_text(encoding="utf-8")
