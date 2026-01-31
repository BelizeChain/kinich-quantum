"""
Quantum loss functions for QML training.

Implements loss functions tailored for quantum machine learning.

Author: Kinich Quantum Team
License: MIT
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class QuantumCrossEntropy:
    """
    Cross-entropy loss for quantum classification.
    
    Computes -sum(y_true * log(y_pred)) with quantum measurement outputs.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize cross-entropy loss.
        
        Args:
            epsilon: Small constant to avoid log(0)
        """
        self.epsilon = epsilon
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: True labels [batch_size, num_classes]
            y_pred: Predicted probabilities [batch_size, num_classes]
            
        Returns:
            Loss value
        """
        # Clip predictions to avoid numerical issues
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute cross-entropy
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return float(loss)
    
    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Gradient [batch_size, num_classes]
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        gradient = -(y_true / y_pred) / len(y_true)
        return gradient


class QuantumMSE:
    """
    Mean Squared Error loss for quantum regression.
    
    Computes mean((y_true - y_pred)^2) with quantum outputs.
    """
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute MSE loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        loss = np.mean((y_true - y_pred) ** 2)
        return float(loss)
    
    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Gradient
        """
        gradient = 2 * (y_pred - y_true) / len(y_true)
        return gradient


class FidelityLoss:
    """
    Quantum state fidelity loss.
    
    Measures similarity between quantum states:
    F(ρ, σ) = Tr(√(√ρ σ √ρ))²
    
    For pure states: F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²
    """
    
    def __call__(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> float:
        """
        Compute fidelity loss (1 - fidelity).
        
        Args:
            state1: Quantum state 1 (probabilities)
            state2: Quantum state 2 (probabilities)
            
        Returns:
            Loss value (lower is better)
        """
        # For probability distributions, use classical fidelity
        # F = (sum sqrt(p_i * q_i))^2
        fidelity = np.sum(np.sqrt(state1 * state2)) ** 2
        
        # Return 1 - fidelity as loss
        loss = 1.0 - fidelity
        
        return float(loss)
    
    def gradient(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient (numerical approximation).
        
        Args:
            state1: Quantum state 1
            state2: Quantum state 2
            
        Returns:
            Gradient w.r.t. state1
        """
        # Numerical gradient
        epsilon = 1e-7
        gradient = np.zeros_like(state1)
        
        base_loss = self(state1, state2)
        
        for i in range(len(state1)):
            state1_plus = state1.copy()
            state1_plus[i] += epsilon
            loss_plus = self(state1_plus, state2)
            
            gradient[i] = (loss_plus - base_loss) / epsilon
        
        return gradient


class QuantumHingeLoss:
    """
    Hinge loss for quantum SVM.
    
    L = max(0, 1 - y * f(x))
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize hinge loss.
        
        Args:
            margin: Margin parameter
        """
        self.margin = margin
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute hinge loss.
        
        Args:
            y_true: True labels (+1 or -1)
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # Convert labels to +1/-1 if needed
        y_true = np.where(y_true > 0, 1, -1)
        
        # Hinge loss
        loss = np.mean(np.maximum(0, self.margin - y_true * y_pred))
        
        return float(loss)
    
    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient.
        
        Args:
            y_true: True labels
            y_pred: Predicted values
            
        Returns:
            Gradient
        """
        y_true = np.where(y_true > 0, 1, -1)
        
        # Gradient is -y where hinge is active
        gradient = np.where(
            self.margin - y_true * y_pred > 0,
            -y_true / len(y_true),
            0
        )
        
        return gradient


class QuantumContrastiveLoss:
    """
    Contrastive loss for quantum metric learning.
    
    Learns embeddings where similar samples are close,
    dissimilar samples are far apart.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        self.margin = margin
    
    def __call__(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        is_similar: np.ndarray
    ) -> float:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embeddings [batch_size, dim]
            embedding2: Second embeddings [batch_size, dim]
            is_similar: Binary labels (1=similar, 0=dissimilar)
            
        Returns:
            Loss value
        """
        # Compute distances
        distances = np.linalg.norm(embedding1 - embedding2, axis=1)
        
        # Loss for similar pairs
        similar_loss = is_similar * (distances ** 2)
        
        # Loss for dissimilar pairs
        dissimilar_loss = (1 - is_similar) * np.maximum(0, self.margin - distances) ** 2
        
        # Total loss
        loss = np.mean(similar_loss + dissimilar_loss)
        
        return float(loss)
