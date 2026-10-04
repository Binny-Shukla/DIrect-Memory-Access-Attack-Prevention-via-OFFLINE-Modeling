from dma_guard import action_id


class block_halt_handler:
    """Stores revoked device sessions; OS driver integration belongs here later."""

    def __init__(self): self.revoked_devices = set()
    def apply(self, device_id, action):
        if action in {action_id.block, action_id.throttle_halt}: self.revoked_devices.add(device_id)
        return device_id in self.revoked_devices
