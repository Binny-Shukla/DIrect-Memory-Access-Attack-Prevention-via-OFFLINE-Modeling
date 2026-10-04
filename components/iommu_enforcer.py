from dma_guard import action_id


class iommu_dma_remapping_enforcer:
    """Adapter boundary for a signed OS driver; this prototype records mappings only."""

    def __init__(self): self.mappings = {}
    def map_region(self, device_id, region_name): self.mappings.setdefault(device_id, set()).add(region_name)
    def enforce(self, device_id, region_name):
        return action_id.allow if region_name in self.mappings.get(device_id, set()) else action_id.block
