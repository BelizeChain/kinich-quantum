"""
Advanced Quantum Feature Maps.

Implements sophisticated quantum encoding strategies beyond basic ZZ and Pauli maps.

Key Features:
- IQP (Instantaneous Quantum Polynomial) encoding
- Custom feature maps with user-defined gates
- Multi-dimensional encoding strategies
- Adaptive feature maps

References:
- Havlíček et al. "Supervised learning with quantum-enhanced feature spaces" (2019)
- Lloyd et al. "Quantum embeddings for machine learning" (2020)

Author: Kinich Quantum Team
License: MIT
"""

import logging
from typing import Optional, List, Callable
import numpy as np

logger = logging.getLogger(__name__)


class IQPFeatureMap:
    """
    Instantaneous Quantum Polynomial (IQP) Feature Map.
    
    Encodes data into quantum states using diagonal unitaries,
    creating non-classical correlations via controlled-Z gates.
    
    Circuit structure:
    1. Hadamard layer (superposition)
    2. Data encoding with RZ gates
    3. Entangling layer with CZ gates
    4. Repeat for 'reps' layers
    """
    
    def __init__(
        self,
        num_features: int,
        reps: int = 2,
        entanglement: str = "full",
        data_scaling: float = 2.0
    ):
        """
        Initialize IQP feature map.
        
        Args:
            num_features: Number of features (qubits)
            reps: Number of repetitions
            entanglement: Entanglement pattern ("full", "linear", "circular")
            data_scaling: Scaling factor for data encoding
        """
        self.num_features = num_features
        self.reps = reps
        self.entanglement = entanglement
        self.data_scaling = data_scaling
        
        # Build entanglement pairs
        self.entangling_pairs = self._get_entangling_pairs()
        
        logger.info(
            f"Initialized IQPFeatureMap: {num_features} features, "
            f"{reps} reps, {entanglement} entanglement"
        )
    
    def _get_entangling_pairs(self) -> List[tuple]:
        """Get pairs of qubits to entangle."""
        pairs = []
        
        if self.entanglement == "full":
            # All-to-all connectivity
            for i in range(self.num_features):
                for j in range(i + 1, self.num_features):
                    pairs.append((i, j))
        
        elif self.entanglement == "linear":
            # Nearest-neighbor only
            for i in range(self.num_features - 1):
                pairs.append((i, i + 1))
        
        elif self.entanglement == "circular":
            # Nearest-neighbor + wrap-around
            for i in range(self.num_features - 1):
                pairs.append((i, i + 1))
            if self.num_features > 2:
                pairs.append((self.num_features - 1, 0))
        
        else:
            raise ValueError(f"Unknown entanglement: {self.entanglement}")
        
        return pairs
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode data into quantum state (mock implementation).
        
        In practice, this would build and execute a quantum circuit.
        For now, returns classical approximation.
        
        Args:
            x: Input features [num_features]
            
        Returns:
            Encoded state vector (mock)
        """
        # Mock: return scaled and transformed features
        encoded = np.zeros(2 ** self.num_features)
        
        # Simulate IQP encoding
        # Phase encoding with polynomial interactions
        phases = np.zeros(2 ** self.num_features)
        
        for i in range(2 ** self.num_features):
            # Get binary representation
            bits = [(i >> j) & 1 for j in range(self.num_features)]
            
            # Single-qubit phases
            phase = sum(x[j] * bits[j] for j in range(self.num_features))
            
            # Two-qubit interaction phases
            for qi, qj in self.entangling_pairs:
                phase += x[qi] * x[qj] * bits[qi] * bits[qj]
            
            phases[i] = phase * self.data_scaling
        
        # Apply phases to uniform superposition
        encoded = np.exp(1j * phases) / np.sqrt(2 ** self.num_features)
        
        return np.abs(encoded) ** 2  # Return probabilities
    
    def get_circuit_depth(self) -> int:
        """Get circuit depth."""
        # Hadamard + RZ + CZ layers per rep
        return 3 * self.reps


class CustomFeatureMap:
    """
    Custom feature map with user-defined gates and structure.
    
    Allows flexible definition of encoding strategies.
    """
    
    def __init__(
        self,
        num_features: int,
        encoding_fn: Callable[[np.ndarray], np.ndarray],
        reps: int = 1
    ):
        """
        Initialize custom feature map.
        
        Args:
            num_features: Number of features
            encoding_fn: Custom encoding function
            reps: Number of repetitions
        """
        self.num_features = num_features
        self.encoding_fn = encoding_fn
        self.reps = reps
        
        logger.info(
            f"Initialized CustomFeatureMap: {num_features} features, "
            f"{reps} reps"
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode data using custom function."""
        # Apply custom encoding
        encoded = self.encoding_fn(x)
        
        # Apply repetitions
        for _ in range(self.reps - 1):
            encoded = self.encoding_fn(encoded)
        
        return encoded


class AmplitudeEncoding:
    """
    Amplitude encoding: directly encode data into amplitudes.
    
    More efficient than basis encoding but requires normalization.
    Encodes 2^n values into n qubits.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize amplitude encoding.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.num_amplitudes = 2 ** num_qubits
        
        logger.info(
            f"Initialized AmplitudeEncoding: {num_qubits} qubits, "
            f"{self.num_amplitudes} amplitudes"
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode data into quantum amplitudes.
        
        Args:
            x: Input data (will be padded/truncated to 2^n)
            
        Returns:
            Normalized amplitude vector
        """
        # Pad or truncate to fit
        if len(x) < self.num_amplitudes:
            x = np.pad(x, (0, self.num_amplitudes - len(x)))
        elif len(x) > self.num_amplitudes:
            x = x[:self.num_amplitudes]
        
        # Normalize
        norm = np.linalg.norm(x)
        if norm > 0:
            amplitudes = x / norm
        else:
            amplitudes = np.zeros_like(x)
            amplitudes[0] = 1.0
        
        return amplitudes


class AngleEncoding:
    """
    Angle encoding: encode data as rotation angles.
    
    Maps classical data to qubit rotation angles.
    Each feature controls one qubit rotation.
    """
    
    def __init__(
        self,
        num_features: int,
        rotation_axis: str = "Y"
    ):
        """
        Initialize angle encoding.
        
        Args:
            num_features: Number of features (qubits)
            rotation_axis: Rotation axis ("X", "Y", or "Z")
        """
        self.num_features = num_features
        self.rotation_axis = rotation_axis
        
        logger.info(
            f"Initialized AngleEncoding: {num_features} features, "
            f"R{rotation_axis} gates"
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode data using rotation angles.
        
        Args:
            x: Input features [num_features]
            
        Returns:
            Encoded state (mock)
        """
        # Mock: return rotation-based encoding
        # In practice, would apply R_Y(θ) gates
        encoded = np.zeros(2 ** self.num_features)
        
        for i in range(2 ** self.num_features):
            # Get binary representation
            bits = [(i >> j) & 1 for j in range(self.num_features)]
            
            # Compute amplitude based on rotations
            amplitude = 1.0
            for j in range(self.num_features):
                if bits[j] == 0:
                    amplitude *= np.cos(x[j] / 2)
                else:
                    amplitude *= np.sin(x[j] / 2)
            
            encoded[i] = amplitude
        
        # Normalize
        norm = np.linalg.norm(encoded)
        if norm > 0:
            encoded /= norm
        
        return np.abs(encoded) ** 2  # Return probabilities


class AdaptiveFeatureMap:
    """
    Adaptive feature map that adjusts based on data distribution.
    
    Learns optimal encoding parameters during training.
    """
    
    def __init__(
        self,
        num_features: int,
        base_map: str = "IQP",
        learning_rate: float = 0.01
    ):
        """
        Initialize adaptive feature map.
        
        Args:
            num_features: Number of features
            base_map: Base feature map type
            learning_rate: Learning rate for adaptation
        """
        self.num_features = num_features
        self.base_map_type = base_map
        self.learning_rate = learning_rate
        
        # Create base map
        if base_map == "IQP":
            self.base_map = IQPFeatureMap(num_features)
        else:
            self.base_map = AngleEncoding(num_features)
        
        # Learnable parameters
        self.scale_params = np.ones(num_features)
        self.shift_params = np.zeros(num_features)
        
        logger.info(
            f"Initialized AdaptiveFeatureMap: {num_features} features, "
            f"base={base_map}"
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode with adaptive scaling and shifting.
        
        Args:
            x: Input features
            
        Returns:
            Encoded state
        """
        # Apply learned transformations
        x_transformed = x * self.scale_params + self.shift_params
        
        # Encode with base map
        encoded = self.base_map.encode(x_transformed)
        
        return encoded
    
    def update_parameters(
        self,
        gradient_scale: np.ndarray,
        gradient_shift: np.ndarray
    ):
        """
        Update adaptive parameters.
        
        Args:
            gradient_scale: Gradient w.r.t. scale parameters
            gradient_shift: Gradient w.r.t. shift parameters
        """
        self.scale_params -= self.learning_rate * gradient_scale
        self.shift_params -= self.learning_rate * gradient_shift
    
    def get_parameters(self) -> dict:
        """Get current parameters."""
        return {
            'scale': self.scale_params.copy(),
            'shift': self.shift_params.copy()
        }
