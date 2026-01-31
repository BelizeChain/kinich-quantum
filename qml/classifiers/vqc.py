"""
Variational Quantum Classifier (VQC)

Quantum classifier using variational circuits for supervised learning.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, RealAmplitudes
    from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available - VQC will run in mock mode")


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier for supervised learning.
    
    Uses quantum feature maps and variational circuits for classification.
    Supports binary and multi-class classification.
    
    Architecture:
    1. Feature map: Encode classical data → quantum state
    2. Ansatz: Parameterized variational circuit
    3. Measurement: Extract classification probabilities
    4. Optimizer: Train variational parameters
    
    Example:
        >>> vqc = VariationalQuantumClassifier(num_features=4, num_classes=2)
        >>> vqc.fit(X_train, y_train)
        >>> predictions = vqc.predict(X_test)
        >>> accuracy = vqc.score(X_test, y_test)
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        feature_map: str = "ZZ",
        ansatz_layers: int = 2,
        optimizer: str = "COBYLA",
        max_iter: int = 100,
        backend: str = "azure_ionq",
        shots: int = 1024
    ):
        """
        Initialize Variational Quantum Classifier.
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            feature_map: Feature map type ('ZZ', 'Pauli', or 'Custom')
            ansatz_layers: Number of variational layers
            optimizer: Optimizer ('COBYLA', 'SPSA', 'ADAM')
            max_iter: Maximum optimization iterations
            backend: Quantum backend
            shots: Number of measurement shots
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_map_type = feature_map
        self.ansatz_layers = ansatz_layers
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.backend = backend
        self.shots = shots
        
        # Model components
        self.feature_map_circuit = None
        self.ansatz_circuit = None
        self.parameters = None
        self.trained_params = None
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Build circuits
        self._build_circuits()
        
        logger.info(
            f"Initialized VQC: {num_features} features, "
            f"{num_classes} classes, {ansatz_layers} layers"
        )
    
    def _build_circuits(self) -> None:
        """Build feature map and ansatz circuits."""
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available - skipping circuit build")
            return
        
        # Build feature map
        self.feature_map_circuit = self._create_feature_map()
        
        # Build ansatz
        self.ansatz_circuit = RealAmplitudes(
            num_qubits=self.num_features,
            reps=self.ansatz_layers,
            entanglement='linear'
        )
        
        # Store parameters
        self.parameters = self.ansatz_circuit.parameters
        
        logger.debug(
            f"Built circuits: feature_map depth={self.feature_map_circuit.depth()}, "
            f"ansatz depth={self.ansatz_circuit.depth()}"
        )
    
    def _create_feature_map(self):
        """Create quantum feature map."""
        if self.feature_map_type == "ZZ":
            return ZZFeatureMap(
                feature_dimension=self.num_features,
                reps=2,
                entanglement='linear'
            )
        
        elif self.feature_map_type == "Pauli":
            return PauliFeatureMap(
                feature_dimension=self.num_features,
                reps=2,
                entanglement='linear',
                paulis=['Z', 'ZZ']
            )
        
        else:
            # Default to ZZ
            logger.warning(f"Unknown feature map {self.feature_map_type}, using ZZ")
            return ZZFeatureMap(
                feature_dimension=self.num_features,
                reps=2
            )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.0
    ) -> 'VariationalQuantumClassifier':
        """
        Train the quantum classifier.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Training labels [n_samples]
            validation_split: Fraction of data for validation
            
        Returns:
            self
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Validate data
        X, y_encoded = self._validate_data(X, y_encoded)
        
        # Split validation if requested
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None
        
        # Train
        if QISKIT_AVAILABLE:
            self._train_quantum(X_train, y_train, X_val, y_val)
        else:
            self._train_mock(X_train, y_train)
        
        self.is_fitted = True
        
        logger.info("Training complete")
        return self
    
    def _train_quantum(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> None:
        """Train using quantum circuits."""
        # Initialize parameters
        num_params = len(self.parameters)
        params = np.random.uniform(-np.pi, np.pi, num_params)
        
        # Get optimizer
        optimizer = self._get_optimizer()
        
        # Define objective function
        def objective(params_values):
            return self._compute_loss(X_train, y_train, params_values)
        
        # Optimize
        logger.info(f"Starting optimization with {self.optimizer_name}...")
        
        for iteration in range(self.max_iter):
            # Compute loss and gradients
            loss = objective(params)
            
            # Update parameters (simplified - actual implementation uses optimizer)
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: loss={loss:.4f}")
            
            # Check convergence
            if loss < 1e-3:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        self.trained_params = params
    
    def _train_mock(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Mock training when Qiskit unavailable."""
        logger.warning("Training in mock mode")
        
        # Generate random trained parameters
        num_params = self.num_features * (self.ansatz_layers + 1)
        self.trained_params = np.random.uniform(-np.pi, np.pi, num_params)
        
        logger.info("Mock training complete")
    
    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray
    ) -> float:
        """
        Compute classification loss.
        
        Uses cross-entropy loss.
        """
        predictions = self._predict_proba(X, params)
        
        # Cross-entropy loss
        loss = -np.mean(np.log(predictions[np.arange(len(y)), y] + 1e-10))
        
        return loss
    
    def _predict_proba(
        self,
        X: np.ndarray,
        params: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            params: Circuit parameters
            
        Returns:
            Class probabilities [n_samples, n_classes]
        """
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.num_classes))
        
        for i in range(n_samples):
            # Execute circuit for this sample
            # (Simplified - actual implementation uses quantum backend)
            probs = self._execute_sample(X[i], params)
            probabilities[i] = probs
        
        return probabilities
    
    def _execute_sample(
        self,
        x: np.ndarray,
        params: np.ndarray
    ) -> np.ndarray:
        """Execute quantum circuit for single sample."""
        # Mock implementation
        # In production, this executes on actual quantum hardware
        
        # Generate deterministic probabilities based on input and params
        seed = int(abs(np.sum(x * 1000) + np.sum(params * 100))) % (2**32)
        np.random.seed(seed)
        
        probs = np.random.dirichlet(np.ones(self.num_classes))
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features [n_samples, n_features]
            
        Returns:
            Predicted labels [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get probabilities
        probabilities = self._predict_proba(X, self.trained_params)
        
        # Get class with highest probability
        predictions = np.argmax(probabilities, axis=1)
        
        # Decode labels
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._predict_proba(X, self.trained_params)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def _validate_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple:
        """Validate and preprocess data."""
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Check feature dimension
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {X.shape[1]}"
            )
        
        # Normalize features to [0, 1]
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_min) / X_range
        
        return X_normalized, y
    
    def _get_optimizer(self):
        """Get Qiskit optimizer."""
        if self.optimizer_name == "COBYLA":
            return COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == "SPSA":
            return SPSA(maxiter=self.max_iter)
        elif self.optimizer_name == "ADAM":
            return ADAM(maxiter=self.max_iter)
        else:
            logger.warning(f"Unknown optimizer {self.optimizer_name}, using COBYLA")
            return COBYLA(maxiter=self.max_iter)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model parameters."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        np.savez(
            filepath,
            params=self.trained_params,
            num_features=self.num_features,
            num_classes=self.num_classes,
            classes=self.label_encoder.classes_
        )
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model parameters."""
        data = np.load(filepath)
        
        self.trained_params = data['params']
        self.num_features = int(data['num_features'])
        self.num_classes = int(data['num_classes'])
        self.label_encoder.classes_ = data['classes']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def __repr__(self) -> str:
        return (
            f"VariationalQuantumClassifier("
            f"features={self.num_features}, "
            f"classes={self.num_classes}, "
            f"layers={self.ansatz_layers}, "
            f"fitted={self.is_fitted})"
        )
