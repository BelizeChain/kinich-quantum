"""
ZZ Feature Map

Encodes classical features using ZZ interactions between qubits.
"""

import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap as QiskitZZFeatureMap
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ZZFeatureMap:
    """
    ZZ Feature Map for quantum encoding.
    
    Encodes classical data x into quantum state using:
    U(x) = exp(-i ∑_{i,j} φ_{ij}(x) Z_i Z_j)
    
    where φ_{ij}(x) = (π - x_i)(π - x_j)
    
    Features:
    - Entangles qubits through ZZ interactions
    - Preserves feature relationships
    - Supports multiple repetitions
    
    Example:
        >>> feature_map = ZZFeatureMap(num_features=4, reps=2)
        >>> quantum_state = feature_map.encode(classical_data)
    """
    
    def __init__(
        self,
        num_features: int,
        reps: int = 2,
        entanglement: str = 'linear',
        insert_barriers: bool = False
    ):
        """
        Initialize ZZ Feature Map.
        
        Args:
            num_features: Number of classical features (= number of qubits)
            reps: Number of repetitions of the encoding
            entanglement: Entanglement pattern ('linear', 'full', 'circular')
            insert_barriers: Add barriers for visualization
        """
        self.num_features = num_features
        self.reps = reps
        self.entanglement = entanglement
        self.insert_barriers = insert_barriers
        
        # Build circuit
        self.circuit = self._build_circuit()
        
        logger.info(
            f"Initialized ZZFeatureMap: {num_features} features, "
            f"{reps} reps, {entanglement} entanglement"
        )
    
    def _build_circuit(self) -> Optional['QuantumCircuit']:
        """Build the feature map circuit."""
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available - feature map in mock mode")
            return None
        
        # Use Qiskit's built-in ZZ feature map
        circuit = QiskitZZFeatureMap(
            feature_dimension=self.num_features,
            reps=self.reps,
            entanglement=self.entanglement,
            insert_barriers=self.insert_barriers
        )
        
        logger.debug(f"Built ZZFeatureMap circuit with depth {circuit.depth()}")
        return circuit
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode classical data into quantum state.
        
        Args:
            data: Classical feature vector [n_features] or batch [n_samples, n_features]
            
        Returns:
            Encoded quantum state parameters
        """
        # Validate input
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {data.shape[1]}"
            )
        
        # Normalize to [0, 2π]
        encoded = self._normalize_features(data)
        
        return encoded
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 2π] range for quantum encoding.
        
        Args:
            data: Raw feature data
            
        Returns:
            Normalized features
        """
        # Min-max normalization to [0, 1]
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1  # Avoid division by zero
        
        normalized = (data - data_min) / data_range
        
        # Scale to [0, 2π]
        encoded = normalized * 2 * np.pi
        
        return encoded
    
    def get_entanglement_pattern(self) -> List[List[int]]:
        """
        Get the qubit entanglement pattern.
        
        Returns:
            List of qubit pairs that are entangled
        """
        pairs = []
        
        if self.entanglement == 'linear':
            # Linear: (0,1), (1,2), (2,3), ...
            for i in range(self.num_features - 1):
                pairs.append([i, i + 1])
        
        elif self.entanglement == 'full':
            # Full: All pairs
            for i in range(self.num_features):
                for j in range(i + 1, self.num_features):
                    pairs.append([i, j])
        
        elif self.entanglement == 'circular':
            # Circular: Linear + (last, first)
            for i in range(self.num_features - 1):
                pairs.append([i, i + 1])
            pairs.append([self.num_features - 1, 0])
        
        return pairs
    
    def get_num_parameters(self) -> int:
        """Get number of parameters in the feature map."""
        return self.num_features * self.reps
    
    def visualize(self) -> str:
        """
        Get text representation of circuit.
        
        Returns:
            Circuit diagram as string
        """
        if self.circuit is None:
            return "Circuit not available (Qiskit not installed)"
        
        return str(self.circuit.draw('text'))
    
    def __repr__(self) -> str:
        return (
            f"ZZFeatureMap("
            f"features={self.num_features}, "
            f"reps={self.reps}, "
            f"entanglement='{self.entanglement}')"
        )
