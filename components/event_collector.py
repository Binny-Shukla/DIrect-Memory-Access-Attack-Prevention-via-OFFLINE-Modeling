from dma_guard import dma_event


class dma_event_collector:
    """Normalizes raw driver telemetry into validated DMA events."""

    def collect(self, payload):
        return dma_event(
            device_id = str(payload['device_id']), address = int(payload['address']),
            size = int(payload['size']), direction = payload['direction'],
            privilege = payload.get('privilege', 'dma_buffer'),
            system_load = float(payload.get('system_load', 0.0))
        )
