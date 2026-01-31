"""
Quantum Feature Maps

Encode classical data into quantum states.
"""

from kinich.qml.feature_maps.zz_feature_map import ZZFeatureMap
from kinich.qml.feature_maps.pauli_feature_map import PauliFeatureMapEncoder
from kinich.qml.feature_maps.advanced_maps import (
    IQPFeatureMap,
    CustomFeatureMap,
    AmplitudeEncoding,
    AngleEncoding,
    AdaptiveFeatureMap
)

__all__ = [
    'ZZFeatureMap',
    'PauliFeatureMapEncoder',
    'IQPFeatureMap',
    'CustomFeatureMap',
    'AmplitudeEncoding',
    'AngleEncoding',
    'AdaptiveFeatureMap'
]
