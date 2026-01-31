"""
QML Training Infrastructure.

Provides optimizers, loss functions, and training utilities for quantum ML.

Author: Kinich Quantum Team
License: MIT
"""

from .optimizers import SPSAOptimizer, COBYLAOptimizer, QuantumAdam
from .losses import QuantumCrossEntropy, QuantumMSE, FidelityLoss
from .trainer import QMLTrainer

__all__ = [
    'SPSAOptimizer',
    'COBYLAOptimizer',
    'QuantumAdam',
    'QuantumCrossEntropy',
    'QuantumMSE',
    'FidelityLoss',
    'QMLTrainer'
]
