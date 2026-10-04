from dma_guard import memory_region


class memory_map_policy_engine:
    """Resolves physical ranges and device access permissions."""

    def __init__(self, regions): self.regions = {item.name: item for item in regions}
    def resolve(self, address, size):
        return next((item for item in self.regions.values() if item.contains(address, size)), None)
    def permitted(self, profile, region, direction):
        return bool(region and region.name in profile.allowed_regions and direction in profile.allowed_directions)
