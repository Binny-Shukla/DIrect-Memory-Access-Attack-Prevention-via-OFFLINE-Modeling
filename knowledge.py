"""Small local RAG-like knowledge retrieval for offline analysis.

The live controller uses policy_cache.  This module is intentionally outside
the hot path: it helps analysts find device profiles and attack knowledge.
"""

from dataclasses import dataclass
import math
import re


def tokenize(text):
    return set(re.findall(r'[a-z0-9_]+', text.lower()))


@dataclass(frozen = True)
class knowledge_record:
    record_id: str
    text: str
    tags: list[str]


class knowledge_layer:

    def __init__(self, records):
        self.records = list(records)
        self.vectors = [tokenize(item.text + ' ' + ' '.join(item.tags)) for item in self.records]

    def retrieve(self, query, limit = 3):
        query_vector = tokenize(query)
        scored = []
        for record, vector in zip(self.records, self.vectors):
            union = len(query_vector | vector)
            score = len(query_vector & vector) / union if union else 0.0
            scored.append((score, record))
        return [record for score, record in sorted(scored, reverse = True, key = lambda item: item[0])[:limit] if score > 0]


default_knowledge = knowledge_layer([
    knowledge_record('gpu-framebuffer', 'A GPU may DMA only into its assigned framebuffer mapping.', ['gpu', 'framebuffer', 'safe']),
    knowledge_record('nic-packet-buffer', 'A NIC may DMA packet data only into approved packet buffers.', ['nic', 'packet', 'safe']),
    knowledge_record('kernel-stack', 'DMA writes to kernel stacks are a critical memory isolation violation.', ['kernel_stack', 'block', 'attack']),
    knowledge_record('thunderbolt-dma', 'Untrusted hot-plug DMA peripherals require IOMMU isolation before authorization.', ['thunderbolt', 'usb', 'iommu', 'attack'])
])
