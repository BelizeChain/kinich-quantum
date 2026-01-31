"""
Quantum Neural Network Base Class

Foundation for all quantum neural network architectures in Kinich.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, ParameterVector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available - QNN will run in mock mode")


class QuantumNeuralNetwork(ABC):
    """
    Base class for Quantum Neural Networks.
    
    Provides common functionality for:
    - Parameterized quantum circuits
    - Forward/backward propagation
    - Parameter management
    - Multi-backend support
    
    Example:
        >>> qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
        >>> output = qnn.forward(input_data)
        >>> grads = qnn.backward(loss_gradient)
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        backend: str = "azure_ionq",
        shots: int = 1024,
        enable_gradient: bool = True
    ):
        """
        Initialize Quantum Neural Network.
        
        Args:
            num_qubits: Number of qubits in circuit
            num_layers: Number of variational layers
            backend: Quantum backend to use
            shots: Number of measurement shots
            enable_gradient: Enable gradient computation
        """
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available - using mock QNN")
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend = backend
        self.shots = shots
        self.enable_gradient = enable_gradient
        
        # Circuit components
        self.circuit: Optional[QuantumCircuit] = None
        self.parameters: Optional[ParameterVector] = None
        self.param_values: Optional[np.ndarray] = None
        
        # Build circuit
        self._build_circuit()
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(
            f"Initialized QNN: {num_qubits} qubits, "
            f"{num_layers} layers, backend={backend}"
        )
    
    @abstractmethod
    def _build_circuit(self) -> None:
        """
        Build the parameterized quantum circuit.
        
        Must be implemented by subclasses to define circuit architecture.
        """
        pass
    
    def _init_parameters(self) -> None:
        """
        Initialize trainable parameters.
        
        Uses Xavier/He initialization scaled for quantum circuits.
        """
        if not QISKIT_AVAILABLE:
            num_params = self.num_qubits * self.num_layers * 3
            self.param_values = np.random.randn(num_params) * 0.01
            return
        
        if self.parameters is None:
            return
        
        num_params = len(self.parameters)
        
        # Xavier initialization adapted for quantum
        # Parameters are angles, so we use uniform [-π, π]
        self.param_values = np.random.uniform(
            -np.pi, np.pi, size=num_params
        )
        
        logger.debug(f"Initialized {num_params} parameters")
    
    def forward(
        self,
        inputs: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through quantum neural network.
        
        Args:
            inputs: Input data to encode (shape: [batch_size, num_features])
            parameters: Optional parameter values (uses self.param_values if None)
            
        Returns:
            Output predictions (shape: [batch_size, num_outputs])
        """
        if parameters is None:
            parameters = self.param_values
        
        # Validate inputs
        inputs = self._validate_inputs(inputs)
        
        # Execute quantum circuit
        outputs = self._execute_circuit(inputs, parameters)
        
        return outputs
    
    def backward(
        self,
        grad_output: np.ndarray,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass for gradient computation.
        
        Uses parameter shift rule for quantum gradients.
        
        Args:
            grad_output: Gradient from next layer [batch_size, num_outputs]
            inputs: Input data from forward pass
            
        Returns:
            Parameter gradients [num_params]
        """
        if not self.enable_gradient:
            raise ValueError("Gradient computation not enabled")
        
        # Compute gradients using parameter shift rule
        # gradients shape: [num_params, batch_size, num_outputs]
        gradients = self._parameter_shift_gradients(inputs)
        
        # Ensure grad_output has correct shape
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
        
        # Chain rule: sum over batch and outputs
        # param_grads[i] = sum_batch sum_outputs (grad_output * gradients[i])
        param_grads = np.zeros(len(self.param_values))
        for i in range(len(self.param_values)):
            param_grads[i] = np.sum(gradients[i] * grad_output)
        
        return param_grads
    
    def _parameter_shift_gradients(
        self,
        inputs: np.ndarray,
        shift: float = np.pi / 2
    ) -> np.ndarray:
        """
        Compute gradients using parameter shift rule.
        
        For parameter θ_i:
        ∂f/∂θ_i = [f(θ + π/2) - f(θ - π/2)] / 2
        
        Args:
            inputs: Input data
            shift: Parameter shift amount (default: π/2)
            
        Returns:
            Gradient matrix [num_params, batch_size, num_outputs]
        """
        num_params = len(self.param_values)
        batch_size = inputs.shape[0]
        
        # Get output shape by running forward once
        sample_output = self.forward(inputs[:1], self.param_values)
        num_outputs = sample_output.shape[1]
        
        gradients = np.zeros((num_params, batch_size, num_outputs))
        
        for i in range(num_params):
            # Shift parameter up
            params_plus = self.param_values.copy()
            params_plus[i] += shift
            output_plus = self.forward(inputs, params_plus)
            
            # Shift parameter down
            params_minus = self.param_values.copy()
            params_minus[i] -= shift
            output_minus = self.forward(inputs, params_minus)
            
            # Gradient
            gradients[i] = (output_plus - output_minus) / 2
        
        return gradients
    
    @abstractmethod
    def _execute_circuit(
        self,
        inputs: np.ndarray,
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Execute quantum circuit with given inputs and parameters.
        
        Must be implemented by subclasses.
        
        Args:
            inputs: Input data
            parameters: Circuit parameters
            
        Returns:
            Circuit outputs
        """
        pass
    
    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input data.
        
        Args:
            inputs: Raw input data
            
        Returns:
            Validated inputs
        """
        # Ensure 2D array
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Check feature dimension
        if inputs.shape[1] > self.num_qubits:
            logger.warning(
                f"Input features ({inputs.shape[1]}) > num_qubits ({self.num_qubits}). "
                "Truncating features."
            )
            inputs = inputs[:, :self.num_qubits]
        
        return inputs
    
    def update_parameters(
        self,
        gradients: np.ndarray,
        learning_rate: float = 0.01
    ) -> None:
        """
        Update parameters using computed gradients.
        
        Args:
            gradients: Parameter gradients
            learning_rate: Learning rate
        """
        self.param_values -= learning_rate * gradients
        
        # Keep parameters in [-π, π]
        self.param_values = np.mod(
            self.param_values + np.pi, 2 * np.pi
        ) - np.pi
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return len(self.param_values) if self.param_values is not None else 0
    
    def save_parameters(self, filepath: str) -> None:
        """
        Save trained parameters to file.
        
        Args:
            filepath: Path to save parameters
        """
        np.save(filepath, self.param_values)
        logger.info(f"Saved parameters to {filepath}")
    
    def load_parameters(self, filepath: str) -> None:
        """
        Load trained parameters from file.
        
        Args:
            filepath: Path to load parameters from
        """
        self.param_values = np.load(filepath)
        logger.info(f"Loaded parameters from {filepath}")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"qubits={self.num_qubits}, "
            f"layers={self.num_layers}, "
            f"params={self.get_num_parameters()}, "
            f"backend={self.backend})"
        )
