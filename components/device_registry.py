from dma_guard import device_profile


class device_identity_trust_registry:
    """Owns trusted-device registration and capability lookup."""

    def __init__(self, profiles = ()): self.profiles = {item.device_id: item for item in profiles}
    def lookup(self, device_id): return self.profiles.get(device_id)
    def register(self, profile: device_profile): self.profiles[profile.device_id] = profile
