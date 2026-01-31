"""
Nawal-Kinich Quantum Bridge

Connects Nawal's classical neural networks with Kinich's quantum processors.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class NawalQuantumBridge:
    """
    Bridge between Nawal (classical ML) and Kinich (quantum ML).
    
    Workflow:
    1. Nawal extracts classical features from data
    2. Bridge encodes features into quantum format
    3. Kinich processes via quantum neural networks
    4. Bridge decodes quantum results back to classical
    5. Nawal uses quantum-enhanced predictions
    
    Example:
        >>> bridge = NawalQuantumBridge(
        ...     classical_dim=128,
        ...     quantum_dim=8
        ... )
        >>> # Classical features from Nawal
        >>> classical_features = nawal_model.extract_features(data)
        >>> 
        >>> # Quantum processing via Kinich
        >>> quantum_output = bridge.classical_to_quantum(classical_features)
        >>> quantum_result = kinich_qnn.forward(quantum_output)
        >>> 
        >>> # Back to classical for Nawal
        >>> classical_result = bridge.quantum_to_classical(quantum_result)
    """
    
    def __init__(
        self,
        classical_dim: int,
        quantum_dim: int,
        encoding_method: str = "pca",
        decoding_method: str = "linear"
    ):
        """
        Initialize Nawal-Kinich bridge.
        
        Args:
            classical_dim: Dimension of classical features (from Nawal)
            quantum_dim: Dimension of quantum features (for Kinich)
            encoding_method: Method to reduce classical → quantum ('pca', 'linear', 'autoencoder')
            decoding_method: Method to expand quantum → classical ('linear', 'interpolate')
        """
        self.classical_dim = classical_dim
        self.quantum_dim = quantum_dim
        self.encoding_method = encoding_method
        self.decoding_method = decoding_method
        
        # Learned transformation matrices
        self.encoding_matrix = None
        self.decoding_matrix = None
        
        # Initialize transformations
        self._init_transformations()
        
        logger.info(
            f"Initialized NawalQuantumBridge: "
            f"classical_dim={classical_dim}, quantum_dim={quantum_dim}"
        )
    
    def _init_transformations(self) -> None:
        """Initialize encoding/decoding transformations."""
        if self.encoding_method == "linear":
            # Linear projection: classical_dim → quantum_dim
            self.encoding_matrix = self._xavier_init(
                (self.quantum_dim, self.classical_dim)
            )
        
        elif self.encoding_method == "pca":
            # PCA-like initialization (will be trained on data)
            self.encoding_matrix = self._xavier_init(
                (self.quantum_dim, self.classical_dim)
            )
        
        if self.decoding_method == "linear":
            # Linear projection: quantum_dim → classical_dim
            self.decoding_matrix = self._xavier_init(
                (self.classical_dim, self.quantum_dim)
            )
    
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot initialization."""
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)
    
    def classical_to_quantum(
        self,
        classical_features: np.ndarray
    ) -> np.ndarray:
        """
        Encode classical features to quantum-compatible format.
        
        Reduces dimensionality: classical_dim → quantum_dim
        
        Args:
            classical_features: Features from Nawal [batch_size, classical_dim]
            
        Returns:
            Quantum-ready features [batch_size, quantum_dim]
        """
        # Ensure 2D
        if classical_features.ndim == 1:
            classical_features = classical_features.reshape(1, -1)
        
        if classical_features.shape[1] != self.classical_dim:
            raise ValueError(
                f"Expected {self.classical_dim} features, "
                f"got {classical_features.shape[1]}"
            )
        
        # Apply encoding transformation
        if self.encoding_method in ["linear", "pca"]:
            quantum_features = classical_features @ self.encoding_matrix.T
        
        else:
            # Fallback: simple truncation
            quantum_features = classical_features[:, :self.quantum_dim]
        
        # Normalize to [0, 1] for quantum encoding
        quantum_features = self._normalize(quantum_features)
        
        logger.debug(
            f"Encoded {classical_features.shape} → {quantum_features.shape}"
        )
        
        return quantum_features
    
    def quantum_to_classical(
        self,
        quantum_results: np.ndarray
    ) -> np.ndarray:
        """
        Decode quantum results back to classical format.
        
        Expands dimensionality: quantum_dim → classical_dim
        
        Args:
            quantum_results: Output from Kinich QNN [batch_size, quantum_dim]
            
        Returns:
            Classical features [batch_size, classical_dim]
        """
        # Ensure 2D
        if quantum_results.ndim == 1:
            quantum_results = quantum_results.reshape(1, -1)
        
        # Apply decoding transformation
        if self.decoding_method == "linear":
            classical_results = quantum_results @ self.decoding_matrix.T
        
        elif self.decoding_method == "interpolate":
            # Interpolate/upsample
            classical_results = self._interpolate_features(quantum_results)
        
        else:
            # Fallback: zero-padding
            batch_size = quantum_results.shape[0]
            classical_results = np.zeros((batch_size, self.classical_dim))
            classical_results[:, :self.quantum_dim] = quantum_results
        
        logger.debug(
            f"Decoded {quantum_results.shape} → {classical_results.shape}"
        )
        
        return classical_results
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1]."""
        min_val = features.min(axis=1, keepdims=True)
        max_val = features.max(axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        
        return (features - min_val) / range_val
    
    def _interpolate_features(self, features: np.ndarray) -> np.ndarray:
        """Interpolate quantum features to classical dimension."""
        batch_size = features.shape[0]
        
        # Linear interpolation indices
        indices = np.linspace(0, self.quantum_dim - 1, self.classical_dim)
        
        # Interpolate for each sample
        interpolated = np.zeros((batch_size, self.classical_dim))
        for i in range(batch_size):
            interpolated[i] = np.interp(
                indices,
                np.arange(self.quantum_dim),
                features[i]
            )
        
        return interpolated
    
    def fit_encoding(
        self,
        classical_data: np.ndarray,
        method: str = "pca"
    ) -> None:
        """
        Train encoding transformation on classical data.
        
        Args:
            classical_data: Training data from Nawal [n_samples, classical_dim]
            method: Encoding method to fit
        """
        if method == "pca":
            # Fit PCA-based encoding
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=self.quantum_dim)
            pca.fit(classical_data)
            
            self.encoding_matrix = pca.components_
            
            logger.info(
                f"Fitted PCA encoding: "
                f"explained_variance={pca.explained_variance_ratio_.sum():.3f}"
            )
        
        else:
            logger.warning(f"fit_encoding not implemented for {method}")
    
    def get_compression_ratio(self) -> float:
        """Get classical→quantum compression ratio."""
        return self.quantum_dim / self.classical_dim
    
    def __repr__(self) -> str:
        return (
            f"NawalQuantumBridge("
            f"classical={self.classical_dim}, "
            f"quantum={self.quantum_dim}, "
            f"compression={self.get_compression_ratio():.2f})"
        )
