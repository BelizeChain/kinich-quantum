"""
Quantum Classifiers
"""

from kinich.qml.classifiers.vqc import VariationalQuantumClassifier
from kinich.qml.classifiers.qsvm import QSVM, QuantumKernel

# Convenient aliases
VQC = VariationalQuantumClassifier

__all__ = ['VariationalQuantumClassifier', 'VQC', 'QSVM', 'QuantumKernel']
