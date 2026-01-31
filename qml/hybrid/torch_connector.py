"""
PyTorch Integration for Kinich QML

Enables seamless integration with PyTorch models and autograd.
"""

import numpy as np
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class QuantumFunction(Function if TORCH_AVAILABLE else object):
    """
    PyTorch autograd function for quantum neural networks.
    
    Implements forward and backward passes with parameter shift rule.
    """
    
    @staticmethod
    def forward(ctx, input_tensor, qnn_model, param_tensor):
        """
        Forward pass through quantum circuit.
        
        Args:
            ctx: Context for saving variables
            input_tensor: Input features [batch_size, num_features]
            qnn_model: Quantum neural network instance
            param_tensor: QNN parameters
            
        Returns:
            Output tensor [batch_size, num_outputs]
        """
        # Convert to numpy
        input_np = input_tensor.detach().cpu().numpy()
        params_np = param_tensor.detach().cpu().numpy()
        
        # Execute quantum forward pass
        output_np = qnn_model.forward(input_np, params_np)
        
        # Convert back to tensor with float32 dtype
        output_tensor = torch.from_numpy(output_np).float().to(input_tensor.device)
        
        # Save for backward
        ctx.save_for_backward(input_tensor, param_tensor)
        ctx.qnn_model = qnn_model
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using parameter shift rule.
        
        Args:
            ctx: Context with saved variables
            grad_output: Gradient from next layer
            
        Returns:
            Gradients for input, qnn_model (None), and parameters
        """
        input_tensor, param_tensor = ctx.saved_tensors
        qnn_model = ctx.qnn_model
        
        # Convert to numpy
        input_np = input_tensor.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        # Compute quantum gradients via parameter shift
        param_grads_np = qnn_model.backward(grad_output_np, input_np)
        
        # Convert to tensor
        param_grads = torch.from_numpy(param_grads_np).to(param_tensor.device)
        
        # Input gradients (optional - for now return None)
        input_grads = None
        
        return input_grads, None, param_grads


class TorchQuantumNeuralNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch wrapper for Kinich Quantum Neural Networks.
    
    Makes QNNs compatible with PyTorch training pipelines.
    
    Example:
        >>> qnn = TorchQuantumNeuralNetwork(num_qubits=4, num_layers=2)
        >>> x = torch.randn(32, 4)  # batch of 32 samples
        >>> y = qnn(x)
        >>> loss = F.mse_loss(y, target)
        >>> loss.backward()  # Quantum gradients computed!
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        ansatz: str = "RealAmplitudes",
        backend: str = "simulator"
    ):
        """
        Initialize PyTorch-compatible QNN.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers
            ansatz: Ansatz type
            backend: Quantum backend
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TorchQuantumNeuralNetwork")
        
        super().__init__()
        
        # Import Kinich QNN
        from kinich.qml.models.circuit_qnn import CircuitQuantumNeuralNetwork
        
        self.qnn = CircuitQuantumNeuralNetwork(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz=ansatz,
            backend=backend
        )
        
        # Register parameters as PyTorch parameters
        num_params = self.qnn.get_num_parameters()
        self.parameters_tensor = nn.Parameter(
            torch.from_numpy(self.qnn.param_values).float()
        )
        
        logger.info(
            f"Initialized TorchQuantumNeuralNetwork: "
            f"{num_qubits} qubits, {num_params} parameters"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.
        
        Args:
            x: Input tensor [batch_size, num_qubits]
            
        Returns:
            Output tensor [batch_size, 2^num_qubits]
        """
        # Ensure float32 dtype for compatibility
        if x.dtype != torch.float32:
            x = x.float()
        
        # Use custom autograd function
        output = QuantumFunction.apply(x, self.qnn, self.parameters_tensor)
        
        return output
    
    def get_quantum_parameters(self) -> np.ndarray:
        """Get current quantum parameters."""
        return self.parameters_tensor.detach().cpu().numpy()
    
    def set_quantum_parameters(self, params: np.ndarray) -> None:
        """Set quantum parameters."""
        self.parameters_tensor.data = torch.from_numpy(params).float()


class HybridQuantumClassicalModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Hybrid model combining classical and quantum layers.
    
    Example architecture for classification:
        Classical Encoder → Quantum Layer → Classical Decoder
    
    Example:
        >>> model = HybridQuantumClassicalModel(
        ...     input_dim=128,
        ...     quantum_dim=8,
        ...     output_dim=10,
        ...     num_quantum_layers=2
        ... )
        >>> 
        >>> x = torch.randn(32, 128)
        >>> logits = model(x)  # [32, 10]
    """
    
    def __init__(
        self,
        input_dim: int,
        quantum_dim: int,
        output_dim: int,
        num_quantum_layers: int = 2,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize hybrid model.
        
        Args:
            input_dim: Input feature dimension
            quantum_dim: Quantum feature dimension (number of qubits)
            output_dim: Output dimension
            num_quantum_layers: Number of quantum variational layers
            hidden_dim: Hidden dimension for classical layers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, quantum_dim)
        )
        
        # Quantum layer
        self.quantum_layer = TorchQuantumNeuralNetwork(
            num_qubits=quantum_dim,
            num_layers=num_quantum_layers
        )
        
        # Classical decoder
        num_quantum_outputs = 2 ** quantum_dim
        self.decoder = nn.Sequential(
            nn.Linear(num_quantum_outputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        # Ensure float32 dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        # Classical encoding
        x = self.encoder(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Classical decoding
        x = self.decoder(x)
        
        return x
