from pathlib import Path
from dma_guard import build_guard, dma_event


checkpoint = 'checkpoint.pt' if Path('checkpoint.pt').exists() else None
guard, logger = build_guard('policy.json', 'logs/dma_audit.jsonl', checkpoint_path = checkpoint)

print(f"[*] DMA Guard initialized with model checkpoint: {checkpoint or 'Untrained'}\n")

events = [
    # 1. Normal legitimate GPU frame write
    dma_event('gpu-0', 268435456, 4096, 'write', 'dma_buffer', system_load = 0.20),
    # 2. Hard attack: Illegal write directly to protected kernel stack
    dma_event('gpu-0', 4026531840, 4096, 'write', 'kernel_stack', system_load = 0.31),
    # 3. Policy violation: Transfer size exceeds device profile max limit
    dma_event('nic-0', 536870912, 2_000_000, 'write', 'dma_buffer', system_load = 0.52),
    # 4. Subtle anomaly: Approved buffer, but highly unaligned address and anomalous load spike
    dma_event('gpu-0', 268435456 + 1337, 4096, 'write', 'dma_buffer', system_load = 0.96)
]

for event in events:
    result = guard.decide(event)
    logger.log(result)
    print(f'Device: {event.device_id:<6} | Decision: {result.action.name:<13} | Risk: {result.risk:.3f} | Reason: {result.reason}')
