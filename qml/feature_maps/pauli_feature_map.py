"""
Pauli Feature Map

Encodes features using Pauli rotations.
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import PauliFeatureMap
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class PauliFeatureMapEncoder:
    """
    Pauli Feature Map for quantum encoding.
    
    Uses Pauli rotations (X, Y, Z) to encode classical data.
    
    Example:
        >>> feature_map = PauliFeatureMapEncoder(num_features=4, paulis=['Z', 'ZZ'])
        >>> encoded = feature_map.encode(data)
    """
    
    def __init__(
        self,
        num_features: int,
        paulis: List[str] = ['Z', 'ZZ'],
        reps: int = 2,
        entanglement: str = 'linear'
    ):
        """
        Initialize Pauli Feature Map.
        
        Args:
            num_features: Number of features
            paulis: Pauli strings to use
            reps: Number of repetitions
            entanglement: Entanglement pattern
        """
        self.num_features = num_features
        self.paulis = paulis
        self.reps = reps
        self.entanglement = entanglement
        
        self.circuit = self._build_circuit()
    
    def _build_circuit(self):
        """Build Pauli feature map circuit."""
        if not QISKIT_AVAILABLE:
            return None
        
        return PauliFeatureMap(
            feature_dimension=self.num_features,
            paulis=self.paulis,
            reps=self.reps,
            entanglement=self.entanglement
        )
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using Pauli rotations."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Normalize to [0, 2π]
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
        return normalized * 2 * np.pi
