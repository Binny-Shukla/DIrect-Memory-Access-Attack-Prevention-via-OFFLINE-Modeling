from offline_pipeline import rollout_policy, validate_model, validate_policy


class policy_validation_rollout_service:
    """Validates candidate artefacts before atomically activating a policy."""

    def validate(self, policy_path, checkpoint_path = None):
        return {'policy_valid': validate_policy(policy_path), 'model_parameters': validate_model(checkpoint_path)}
    def rollout(self, candidate_path, active_path):
        return rollout_policy(candidate_path, active_path)
