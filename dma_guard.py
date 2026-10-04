"""DMA Guard prototype.

This module is a user-space policy simulator.  It deliberately does not try to
program an IOMMU or block devices directly; a production adapter must be a
signed OS driver using the platform IOMMU APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from time import monotonic_ns
from typing import Iterable
import json
import math

import torch
import torch.nn as nn


# action IDs

class action_id(IntEnum):
    allow = 0
    throttle_halt = 1
    block = 2
    log_alert = 3


@dataclass(frozen = True)
class dma_event:
    device_id: str
    address: int
    size: int
    direction: str                 # read or write, from the device perspective
    privilege: str                 # user, dma_buffer, kernel, kernel_stack
    timestamp_ns: int = field(default_factory = monotonic_ns)
    system_load: float = 0.0


@dataclass(frozen = True)
class memory_region:
    start: int
    end: int                       # exclusive
    privilege: str
    name: str

    def contains(self, address: int, size: int) -> bool:
        return self.start <= address and address + size <= self.end


@dataclass
class device_profile:
    device_id: str
    device_type: str
    trusted: bool
    allowed_regions: list[str]
    allowed_directions: list[str]
    max_transfer_size: int
    max_requests_per_second: int


@dataclass
class decision:
    action: action_id
    reason: str
    risk: float
    event: dma_event


class local_profile_store:

    def __init__(self, profiles: Iterable[device_profile], regions: Iterable[memory_region]):
        self.profiles = {profile.device_id: profile for profile in profiles}
        self.regions = {region.name: region for region in regions}

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding = 'utf-8'))
        profiles = [device_profile(**item) for item in payload['device_profiles']]
        regions = [memory_region(**item) for item in payload['memory_regions']]
        return cls(profiles, regions)


class policy_cache:

    def __init__(self, store: local_profile_store):
        self.store = store

    def profile(self, device_id):
        return self.store.profiles.get(device_id)

    def region_for(self, event: dma_event):
        for region in self.store.regions.values():
            if region.contains(event.address, event.size):
                return region
        return None


class deterministic_rule_engine:

    protected_privileges = {'kernel', 'kernel_stack'}

    def __init__(self, cache: policy_cache):
        self.cache = cache
        self.request_times = {}

    def evaluate(self, event: dma_event):
        profile = self.cache.profile(event.device_id)
        region = self.cache.region_for(event)

        if profile is None or not profile.trusted:
            return action_id.block, 'unknown or untrusted device'
        if event.size <= 0:
            return action_id.block, 'invalid transfer size'
        if region is None:
            return action_id.block, 'address is outside every approved DMA region'
        if region.privilege in self.protected_privileges:
            return action_id.block, 'protected kernel memory is never DMA accessible'
        if region.name not in profile.allowed_regions:
            return action_id.block, 'device is not approved for this memory region'
        if event.direction not in profile.allowed_directions:
            return action_id.block, 'DMA direction is not approved for device'
        if event.size > profile.max_transfer_size:
            return action_id.throttle_halt, 'transfer exceeds device size limit'
        if self._too_frequent(event, profile):
            return action_id.throttle_halt, 'device exceeds request-rate limit'
        return action_id.allow, 'matches deterministic DMA policy'

    def _too_frequent(self, event, profile):
        times = self.request_times.setdefault(event.device_id, [])
        cutoff = event.timestamp_ns - 1_000_000_000
        times[:] = [value for value in times if value >= cutoff]
        times.append(event.timestamp_ns)
        return len(times) > profile.max_requests_per_second


class fast_feature_extractor:

    def __init__(self, cache: policy_cache):
        self.cache = cache

    def extract(self, event: dma_event):
        region = self.cache.region_for(event)
        return torch.tensor([
            math.log2(max(event.size, 1)) / 32.0,
            event.system_load,
            1.0 if event.direction == 'write' else 0.0,
            1.0 if region is None else 0.0,
            1.0 if event.privilege in {'kernel', 'kernel_stack'} else 0.0,
            (event.address % 4096) / 4096.0,
        ], dtype = torch.float32)


class small_risk_model(nn.Module):

    def __init__(self, feature_size = 6, d_model = 64, num_layers = 2, num_heads = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(feature_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, batch_first = True, activation = 'gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 4)
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        x = self.input_projection(features)
        x = self.encoder(x)
        return self.classifier(self.norm(x[:, -1]))


class decision_controller:

    def __init__(self, rules, extractor, model = None, model_threshold = 0.80):
        self.rules = rules
        self.extractor = extractor
        self.model = model
        self.model_threshold = model_threshold

    @torch.inference_mode()
    def decide(self, event: dma_event):
        rule_action, reason = self.rules.evaluate(event)
        if rule_action != action_id.allow:
            return decision(rule_action, reason, 1.0, event)

        if self.model is None:
            return decision(action_id.allow, reason, 0.0, event)
        logits = self.model(self.extractor.extract(event).unsqueeze(0))
        probabilities = torch.softmax(logits, dim = -1)[0]
        suggested = action_id(int(probabilities.argmax()))
        risk = float(1.0 - probabilities[action_id.allow])
        if suggested in {action_id.block, action_id.throttle_halt} and float(probabilities[suggested]) >= self.model_threshold:
            return decision(action_id.log_alert, f'model anomaly recommendation: {suggested.name}', risk, event)
        return decision(action_id.allow, reason, risk, event)


class audit_logger:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)

    def log(self, item: decision):
        payload = asdict(item)
        payload['action'] = int(item.action)
        with self.path.open('a', encoding = 'utf-8') as file:
            file.write(json.dumps(payload) + '\n')


def build_guard(policy_path, log_path, checkpoint_path = None):
    store = local_profile_store.from_json(policy_path)
    cache = policy_cache(store)
    model = small_risk_model()
    if sum(parameter.numel() for parameter in model.parameters()) > 20_000_000:
        raise ValueError('risk model violates the 20M parameter budget')
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu', weights_only = True))
    model.eval()
    return decision_controller(deterministic_rule_engine(cache), fast_feature_extractor(cache), model), audit_logger(log_path)
