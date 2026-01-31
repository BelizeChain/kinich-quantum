"""
Kinich Quantum Machine Learning (QML) Module

Quantum neural networks, classifiers, and hybrid classical-quantum models.
"""

from kinich.qml.models.qnn import QuantumNeuralNetwork
from kinich.qml.classifiers.vqc import VariationalQuantumClassifier
from kinich.qml.feature_maps.zz_feature_map import ZZFeatureMap

__all__ = [
    'QuantumNeuralNetwork',
    'VariationalQuantumClassifier',
    'ZZFeatureMap',
]

__version__ = '1.0.0'
