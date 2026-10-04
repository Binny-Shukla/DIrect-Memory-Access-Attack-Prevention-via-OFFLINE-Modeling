from knowledge import default_knowledge, knowledge_layer, knowledge_record


class dma_rag_knowledge_layer(knowledge_layer):
    """Offline profile/attack retrieval; never used in the DMA hot path."""
    pass


__all__ = ['dma_rag_knowledge_layer', 'knowledge_record', 'default_knowledge']
