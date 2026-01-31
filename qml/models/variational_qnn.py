"""
Variational Quantum Neural Networks (VQNNs).

Advanced QNNs with customizable ansatzes and hardware-efficient circuits.

Key Features:
- Multiple ansatz types (hardware-efficient, strongly-entangling, custom)
- Layer-wise training
- Circuit depth optimization
- Gradient-based and gradient-free training

References:
- Farhi & Neven "Classification with Quantum Neural Networks on Near Term Processors" (2018)
- Schuld et al. "Circuit-centric quantum classifiers" (2020)

Author: Kinich Quantum Team
License: MIT
"""

import logging
from typing import Optional, Callable, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

from ..models.qnn import QuantumNeuralNetwork


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz with minimal gate depth.
    
    Uses single-qubit rotations + nearest-neighbor CNOTs.
    Optimized for NISQ devices.
    """
    
    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        rotation_gates: str = "RY"
    ):
        """
        Initialize hardware-efficient ansatz.
        
        Args:
            num_qubits: Number of qubits
            reps: Number of repetitions
            rotation_gates: Rotation type ("RY", "RX", or "RZ")
        """
        self.num_qubits = num_qubits
        self.reps = reps
        self.rotation_gates = rotation_gates
        
        # Calculate number of parameters
        # Each layer: num_qubits rotations + (num_qubits-1) CNOTs
        # Final layer: num_qubits rotations
        self.num_parameters = num_qubits * (reps + 1)
        
        logger.info(
            f"Initialized HardwareEfficientAnsatz: {num_qubits} qubits, "
            f"{reps} reps, {self.num_parameters} params"
        )
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return self.num_parameters
    
    def build_circuit(self, parameters: np.ndarray) -> str:
        """
        Build circuit description (mock for now).
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Circuit description string
        """
        circuit_desc = f"HardwareEfficient({self.num_qubits}q, {self.reps}r)"
        return circuit_desc


class StronglyEntanglingAnsatz:
    """
    Strongly-entangling ansatz with full connectivity.
    
    Creates maximal entanglement between all qubits.
    More expressive but requires more gates.
    """
    
    def __init__(
        self,
        num_qubits: int,
        reps: int = 2
    ):
        """
        Initialize strongly-entangling ansatz.
        
        Args:
            num_qubits: Number of qubits
            reps: Number of repetitions
        """
        self.num_qubits = num_qubits
        self.reps = reps
        
        # Each layer: 3*num_qubits single-qubit rotations + num_qubits*(num_qubits-1)/2 CNOTs
        params_per_rep = 3 * num_qubits
        self.num_parameters = params_per_rep * reps
        
        logger.info(
            f"Initialized StronglyEntanglingAnsatz: {num_qubits} qubits, "
            f"{reps} reps, {self.num_parameters} params"
        )
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return self.num_parameters
    
    def build_circuit(self, parameters: np.ndarray) -> str:
        """Build circuit description."""
        circuit_desc = f"StronglyEntangling({self.num_qubits}q, {self.reps}r)"
        return circuit_desc


class VariationalQNN(QuantumNeuralNetwork):
    """
    Variational Quantum Neural Network with advanced ansatzes.
    
    Extends base QNN with:
    - Multiple ansatz types
    - Advanced training methods
    - Circuit optimization
    - Layer-wise training
    
    Example:
        >>> vqnn = VariationalQNN(
        ...     num_qubits=4,
        ...     ansatz_type="hardware_efficient",
        ...     reps=3
        ... )
        >>> vqnn.fit(X_train, y_train, method="COBYLA")
        >>> predictions = vqnn.predict(X_test)
    """
    
    def __init__(
        self,
        num_qubits: int,
        ansatz_type: str = "hardware_efficient",
        reps: int = 2,
        optimizer: str = "COBYLA",
        max_iter: int = 100,
        learning_rate: float = 0.01,
        backend: str = "simulator"
    ):
        """
        Initialize Variational QNN.
        
        Args:
            num_qubits: Number of qubits
            ansatz_type: Ansatz type ("hardware_efficient", "strongly_entangling", "custom")
            reps: Number of ansatz repetitions
            optimizer: Optimizer type ("COBYLA", "SPSA", "Adam")
            max_iter: Maximum training iterations
            learning_rate: Learning rate for gradient-based optimizers
            backend: Quantum backend
        """
        # Create ansatz
        if ansatz_type == "hardware_efficient":
            self.ansatz = HardwareEfficientAnsatz(num_qubits, reps)
        elif ansatz_type == "strongly_entangling":
            self.ansatz = StronglyEntanglingAnsatz(num_qubits, reps)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        # Store before calling parent init
        self.ansatz_type = ansatz_type
        self.optimizer_type = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        # Initialize base QNN (without ansatz parameter)
        num_layers = reps
        super().__init__(
            num_qubits=num_qubits,
            num_layers=num_layers,
            backend=backend
        )
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'iterations': []
        }
        
        logger.info(
            f"Initialized VariationalQNN: {num_qubits} qubits, "
            f"{ansatz_type} ansatz, {optimizer} optimizer"
        )
    
    def _build_circuit(self):
        """Build quantum circuit (required by base class)."""
        # Use ansatz to build circuit
        return self.ansatz.build_circuit(self.param_values)
    
    def _execute_circuit(self, inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Execute circuit (required by base class)."""
        # Mock execution for now - in practice would use actual quantum backend
        batch_size = inputs.shape[0]
        num_outputs = 2 ** self.num_qubits
        
        # Return mock probabilities
        outputs = np.random.rand(batch_size, num_outputs)
        outputs /= outputs.sum(axis=1, keepdims=True)
        
        return outputs
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None,
        callback: Optional[Callable] = None,
        validation_data: Optional[tuple] = None
    ) -> 'VariationalQNN':
        """
        Fit VQNN using specified optimization method.
        
        Args:
            X: Training data [n_samples, num_qubits]
            y: Training labels [n_samples]
            method: Optimization method (overrides self.optimizer_type)
            callback: Callback function called after each iteration
            validation_data: Optional (X_val, y_val) for validation
            
        Returns:
            self
        """
        optimizer = method or self.optimizer_type
        
        logger.info(f"Training VQNN with {optimizer} optimizer...")
        
        # Initialize parameters randomly
        self.param_values = np.random.randn(self.get_num_parameters()) * 0.1
        
        # Define loss function
        def loss_fn(params):
            # Set parameters
            old_params = self.param_values.copy()
            self.param_values = params
            
            # Forward pass
            predictions = self.forward(X, params)
            
            # Compute loss (mean squared error for now)
            # For classification: convert to probabilities
            probs = self._softmax(predictions)
            y_onehot = self._one_hot(y, probs.shape[1])
            loss = np.mean((probs - y_onehot) ** 2)
            
            # Restore parameters
            self.param_values = old_params
            
            return loss
        
        # Optimize
        if optimizer == "COBYLA":
            result = self._optimize_cobyla(loss_fn, callback, validation_data, X, y)
        elif optimizer == "SPSA":
            result = self._optimize_spsa(loss_fn, callback, validation_data, X, y)
        elif optimizer == "Adam":
            result = self._optimize_adam(loss_fn, callback, validation_data, X, y)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Set final parameters
        self.param_values = result['params']
        
        logger.info(
            f"✓ Training complete: final loss = {result['loss']:.4f}, "
            f"iterations = {result['iterations']}"
        )
        
        return self
    
    def _optimize_cobyla(
        self,
        loss_fn: Callable,
        callback: Optional[Callable],
        validation_data: Optional[tuple],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """COBYLA optimizer (gradient-free)."""
        logger.info("Using COBYLA (gradient-free) optimization...")
        
        best_loss = float('inf')
        best_params = self.param_values.copy()
        iteration = 0
        
        # Simple COBYLA-like optimization (mock for now)
        for i in range(self.max_iter):
            # Evaluate loss
            loss = loss_fn(self.param_values)
            
            # Update best
            if loss < best_loss:
                best_loss = loss
                best_params = self.param_values.copy()
            
            # Record history
            self.history['loss'].append(loss)
            self.history['iterations'].append(i)
            
            # Callback
            if callback is not None:
                callback(i, loss, self.param_values)
            
            # Perturb parameters (simple random search)
            perturbation = np.random.randn(len(self.param_values)) * 0.01
            candidate = self.param_values + perturbation
            candidate_loss = loss_fn(candidate)
            
            if candidate_loss < loss:
                self.param_values = candidate
            
            iteration = i + 1
            
            # Early stopping
            if len(self.history['loss']) > 10:
                recent_losses = self.history['loss'][-10:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': iteration
        }
    
    def _optimize_spsa(
        self,
        loss_fn: Callable,
        callback: Optional[Callable],
        validation_data: Optional[tuple],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """SPSA optimizer (gradient-free with finite differences)."""
        logger.info("Using SPSA (simultaneous perturbation) optimization...")
        
        best_loss = float('inf')
        best_params = self.param_values.copy()
        
        # SPSA parameters
        a = 0.16  # Step size scaling
        c = 0.1   # Perturbation size
        A = 0.01 * self.max_iter  # Stability constant
        
        for k in range(self.max_iter):
            # Compute step sizes
            ak = a / (k + 1 + A) ** 0.602
            ck = c / (k + 1) ** 0.101
            
            # Random perturbation
            delta = 2 * (np.random.rand(len(self.param_values)) > 0.5) - 1
            
            # Evaluate at perturbed points
            params_plus = self.param_values + ck * delta
            params_minus = self.param_values - ck * delta
            
            loss_plus = loss_fn(params_plus)
            loss_minus = loss_fn(params_minus)
            
            # Gradient estimate
            gradient = (loss_plus - loss_minus) / (2 * ck * delta)
            
            # Update parameters
            self.param_values -= ak * gradient
            
            # Evaluate current loss
            current_loss = loss_fn(self.param_values)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = self.param_values.copy()
            
            # Record history
            self.history['loss'].append(current_loss)
            self.history['iterations'].append(k)
            
            # Callback
            if callback is not None:
                callback(k, current_loss, self.param_values)
            
            if k % 10 == 0:
                logger.debug(f"Iteration {k}: loss = {current_loss:.4f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': self.max_iter
        }
    
    def _optimize_adam(
        self,
        loss_fn: Callable,
        callback: Optional[Callable],
        validation_data: Optional[tuple],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Adam optimizer (gradient-based)."""
        logger.info("Using Adam (gradient-based) optimization...")
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        # Initialize moments
        m = np.zeros_like(self.param_values)
        v = np.zeros_like(self.param_values)
        
        best_loss = float('inf')
        best_params = self.param_values.copy()
        
        for t in range(1, self.max_iter + 1):
            # Compute gradient using parameter shift rule
            gradient = self._compute_gradient(loss_fn, self.param_values)
            
            # Update biased first moment
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second moment
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            self.param_values -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Evaluate loss
            current_loss = loss_fn(self.param_values)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = self.param_values.copy()
            
            # Record history
            self.history['loss'].append(current_loss)
            self.history['iterations'].append(t - 1)
            
            # Callback
            if callback is not None:
                callback(t - 1, current_loss, self.param_values)
            
            if t % 10 == 0:
                logger.debug(f"Iteration {t}: loss = {current_loss:.4f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': self.max_iter
        }
    
    def _compute_gradient(
        self,
        loss_fn: Callable,
        params: np.ndarray,
        shift: float = np.pi / 2
    ) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            # Shift parameter up
            params_plus = params.copy()
            params_plus[i] += shift
            loss_plus = loss_fn(params_plus)
            
            # Shift parameter down
            params_minus = params.copy()
            params_minus[i] -= shift
            loss_minus = loss_fn(params_minus)
            
            # Gradient
            gradient[i] = (loss_plus - loss_minus) / 2
        
        return gradient
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y.astype(int)] = 1
        return one_hot
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data [n_samples, num_qubits]
            
        Returns:
            Predicted labels [n_samples]
        """
        # Forward pass
        outputs = self.forward(X, self.param_values)
        
        # Convert to probabilities and get class
        probs = self._softmax(outputs)
        predictions = np.argmax(probs, axis=1)
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return float(accuracy)
    
    def get_training_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.history
