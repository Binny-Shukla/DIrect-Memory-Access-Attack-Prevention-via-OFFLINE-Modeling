"""Offline-only model/policy validation and atomic rollout helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from dma_guard import small_risk_model


def validate_policy(policy_path):
    payload = json.loads(Path(policy_path).read_text(encoding = 'utf-8'))
    regions = payload.get('memory_regions', [])
    profiles = payload.get('device_profiles', [])
    names = {region['name'] for region in regions}
    if not regions or not profiles:
        raise ValueError('policy requires memory regions and device profiles')
    for region in regions:
        if region['start'] >= region['end']:
            raise ValueError(f"invalid range: {region['name']}")
    for profile in profiles:
        if not set(profile['allowed_regions']).issubset(names):
            raise ValueError(f"unknown region in profile: {profile['device_id']}")
        if profile['max_transfer_size'] <= 0 or profile['max_requests_per_second'] <= 0:
            raise ValueError(f"invalid limits: {profile['device_id']}")
    return True


def validate_model(checkpoint_path = None):
    model = small_risk_model()
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu', weights_only = True))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 20_000_000:
        raise ValueError(f'model has {parameter_count:,} parameters; maximum is 20,000,000')
    with torch.inference_mode():
        output = model(torch.zeros(1, 6))
    if output.shape != (1, 4):
        raise ValueError('model must return four action logits')
    return parameter_count


def rollout_policy(candidate_path, active_path):
    """Validate then replace the active local policy in one filesystem operation."""
    validate_policy(candidate_path)
    candidate = Path(candidate_path)
    active = Path(active_path)
    staged = active.with_suffix('.next.json')
    staged.write_bytes(candidate.read_bytes())
    staged.replace(active)
    return hashlib.sha256(active.read_bytes()).hexdigest()


if __name__ == '__main__':
    print(f'Policy valid: {validate_policy("policy.json")}')
    print(f'Model parameters: {validate_model():,}')
