"""
Quantum-aware optimizers for QML training.

Implements optimizers designed for quantum circuit parameter optimization.

Author: Kinich Quantum Team
License: MIT
"""

import logging
from typing import Callable, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA).
    
    Gradient-free optimizer well-suited for noisy quantum systems.
    Estimates gradient using only 2 function evaluations per iteration.
    
    References:
    - Spall "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation" (1992)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        perturbation: float = 0.1,
        decay_rate: float = 0.602,
        callback: Optional[Callable] = None
    ):
        """
        Initialize SPSA optimizer.
        
        Args:
            max_iter: Maximum iterations
            learning_rate: Initial learning rate (a)
            perturbation: Initial perturbation size (c)
            decay_rate: Decay rate for step sizes
            callback: Optional callback function(iteration, loss, params)
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.decay_rate = decay_rate
        self.callback = callback
        
        # History
        self.history = {
            'loss': [],
            'parameters': [],
            'gradients': []
        }
        
        logger.info(
            f"Initialized SPSAOptimizer: max_iter={max_iter}, "
            f"lr={learning_rate}, pert={perturbation}"
        )
    
    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Minimize loss function.
        
        Args:
            loss_fn: Loss function taking parameters
            initial_params: Initial parameter values
            
        Returns:
            Optimization result dict
        """
        params = initial_params.copy()
        best_params = params.copy()
        best_loss = float('inf')
        
        # SPSA constants
        A = 0.01 * self.max_iter  # Stability constant
        
        for k in range(self.max_iter):
            # Compute step sizes
            ak = self.learning_rate / (k + 1 + A) ** self.decay_rate
            ck = self.perturbation / (k + 1) ** 0.101
            
            # Random perturbation direction
            delta = 2 * (np.random.rand(len(params)) > 0.5) - 1
            
            # Evaluate at perturbed points
            params_plus = params + ck * delta
            params_minus = params - ck * delta
            
            loss_plus = loss_fn(params_plus)
            loss_minus = loss_fn(params_minus)
            
            # Gradient estimate
            gradient = (loss_plus - loss_minus) / (2 * ck * delta)
            
            # Update parameters
            params = params - ak * gradient
            
            # Evaluate current loss
            current_loss = loss_fn(params)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()
            
            # Record history
            self.history['loss'].append(current_loss)
            self.history['parameters'].append(params.copy())
            self.history['gradients'].append(gradient.copy())
            
            # Callback
            if self.callback is not None:
                self.callback(k, current_loss, params)
            
            if k % 10 == 0:
                logger.debug(f"SPSA iteration {k}: loss={current_loss:.6f}")
        
        logger.info(f"✓ SPSA optimization complete: best_loss={best_loss:.6f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': self.max_iter,
            'history': self.history
        }


class COBYLAOptimizer:
    """
    Constrained Optimization BY Linear Approximation (COBYLA).
    
    Gradient-free optimizer using linear approximations.
    Good for quantum optimization with constraints.
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        callback: Optional[Callable] = None
    ):
        """
        Initialize COBYLA optimizer.
        
        Args:
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            callback: Optional callback
        """
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        
        self.history = {
            'loss': [],
            'parameters': []
        }
        
        logger.info(f"Initialized COBYLAOptimizer: max_iter={max_iter}")
    
    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Minimize loss function.
        
        Args:
            loss_fn: Loss function
            initial_params: Initial parameters
            
        Returns:
            Optimization result
        """
        params = initial_params.copy()
        best_params = params.copy()
        best_loss = loss_fn(params)
        
        # Simple COBYLA-like optimization
        step_size = 0.1
        
        for k in range(self.max_iter):
            # Evaluate current loss
            current_loss = loss_fn(params)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()
            
            # Record history
            self.history['loss'].append(current_loss)
            self.history['parameters'].append(params.copy())
            
            # Callback
            if self.callback is not None:
                self.callback(k, current_loss, params)
            
            # Try random perturbations
            best_improvement = 0
            best_direction = None
            
            for _ in range(10):  # Sample directions
                direction = np.random.randn(len(params))
                direction /= np.linalg.norm(direction)
                
                candidate = params + step_size * direction
                candidate_loss = loss_fn(candidate)
                
                improvement = current_loss - candidate_loss
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_direction = direction
            
            # Update if improvement found
            if best_direction is not None and best_improvement > 0:
                params = params + step_size * best_direction
            else:
                # Reduce step size
                step_size *= 0.9
            
            # Convergence check
            if len(self.history['loss']) > 10:
                recent = self.history['loss'][-10:]
                if max(recent) - min(recent) < self.tol:
                    logger.info(f"Converged at iteration {k}")
                    break
            
            if k % 10 == 0:
                logger.debug(f"COBYLA iteration {k}: loss={current_loss:.6f}")
        
        logger.info(f"✓ COBYLA optimization complete: best_loss={best_loss:.6f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': k + 1,
            'history': self.history
        }


class QuantumAdam:
    """
    Quantum-aware Adam optimizer.
    
    Adapts classical Adam for quantum parameter optimization
    using parameter shift gradients.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 100,
        gradient_fn: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ):
        """
        Initialize Quantum Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            max_iter: Maximum iterations
            gradient_fn: Function to compute gradients (uses parameter shift if None)
            callback: Optional callback
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gradient_fn = gradient_fn
        self.callback = callback
        
        self.history = {
            'loss': [],
            'parameters': [],
            'gradients': []
        }
        
        logger.info(
            f"Initialized QuantumAdam: lr={learning_rate}, "
            f"beta1={beta1}, beta2={beta2}"
        )
    
    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Minimize loss function using Adam.
        
        Args:
            loss_fn: Loss function
            initial_params: Initial parameters
            
        Returns:
            Optimization result
        """
        params = initial_params.copy()
        
        # Initialize moments
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        
        best_params = params.copy()
        best_loss = float('inf')
        
        for t in range(1, self.max_iter + 1):
            # Compute gradient
            if self.gradient_fn is not None:
                gradient = self.gradient_fn(params)
            else:
                gradient = self._parameter_shift_gradient(loss_fn, params)
            
            # Update biased first moment
            m = self.beta1 * m + (1 - self.beta1) * gradient
            
            # Update biased second moment
            v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update parameters
            params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Evaluate loss
            current_loss = loss_fn(params)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()
            
            # Record history
            self.history['loss'].append(current_loss)
            self.history['parameters'].append(params.copy())
            self.history['gradients'].append(gradient.copy())
            
            # Callback
            if self.callback is not None:
                self.callback(t - 1, current_loss, params)
            
            if t % 10 == 0:
                logger.debug(f"Adam iteration {t}: loss={current_loss:.6f}")
        
        logger.info(f"✓ Adam optimization complete: best_loss={best_loss:.6f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'iterations': self.max_iter,
            'history': self.history
        }
    
    def _parameter_shift_gradient(
        self,
        loss_fn: Callable[[np.ndarray], float],
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
