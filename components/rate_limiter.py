from time import monotonic_ns


class dma_rate_limiter:
    """Sliding one-second request-rate limiter for throttle decisions."""

    def __init__(self): self.request_times = {}
    def exceeded(self, device_id, maximum, now_ns = None):
        now_ns = now_ns or monotonic_ns()
        items = self.request_times.setdefault(device_id, [])
        items[:] = [item for item in items if item >= now_ns - 1_000_000_000]
        items.append(now_ns)
        return len(items) > maximum
