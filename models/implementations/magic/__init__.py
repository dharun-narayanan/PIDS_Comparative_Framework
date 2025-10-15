"""
MAGIC Model Implementation
Standalone implementation of MAGIC (Masked Graph Representation Learning) for APT Detection.

This is a self-contained version extracted from the original MAGIC repository,
adapted to work with the PIDS Comparative Framework without external dependencies.

Paper: "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning"
USENIX Security 2024
"""

from .gat import GAT, GATConv
from .autoencoder import GMAEModel, build_model
from .loss_func import sce_loss
from .eval import evaluate_entity_level_using_knn

__all__ = [
    'GAT',
    'GATConv',
    'GMAEModel',
    'build_model',
    'sce_loss',
    'evaluate_entity_level_using_knn'
]
