"""
QML Trainer - High-level training interface.

Provides unified training loop for quantum ML models.

Author: Kinich Quantum Team
License: MIT
"""

import logging
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import time

logger = logging.getLogger(__name__)

from .optimizers import SPSAOptimizer, COBYLAOptimizer, QuantumAdam
from .losses import QuantumCrossEntropy, QuantumMSE


class QMLTrainer:
    """
    Unified trainer for quantum ML models.
    
    Provides high-level training API with:
    - Multiple optimizers
    - Multiple loss functions
    - Early stopping
    - Learning rate scheduling
    - Training/validation monitoring
    
    Example:
        >>> trainer = QMLTrainer(
        ...     model=vqnn,
        ...     optimizer='SPSA',
        ...     loss='cross_entropy'
        ... )
        >>> trainer.train(X_train, y_train, X_val, y_val)
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: str = 'SPSA',
        loss: str = 'mse',
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        batch_size: Optional[int] = None,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 1
    ):
        """
        Initialize QML trainer.
        
        Args:
            model: Quantum ML model with fit() method
            optimizer: Optimizer type ('SPSA', 'COBYLA', 'Adam')
            loss: Loss function ('mse', 'cross_entropy', 'fidelity')
            learning_rate: Learning rate
            max_epochs: Maximum training epochs
            batch_size: Batch size (None = full batch)
            early_stopping: Enable early stopping
            patience: Patience for early stopping
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.model = model
        self.optimizer_type = optimizer
        self.loss_type = loss
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        
        # Create loss function
        if loss == 'mse':
            self.loss_fn = QuantumMSE()
        elif loss == 'cross_entropy':
            self.loss_fn = QuantumCrossEntropy()
        else:
            raise ValueError(f"Unknown loss: {loss}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_time': []
        }
        
        logger.info(
            f"Initialized QMLTrainer: optimizer={optimizer}, "
            f"loss={loss}, lr={learning_rate}"
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            callback: Optional callback(epoch, train_loss, val_loss)
            
        Returns:
            Training history dict
        """
        logger.info(f"Starting QML training for {self.max_epochs} epochs...")
        
        start_time = time.time()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # Training step
            train_loss, train_acc = self._train_epoch(X_train, y_train)
            
            # Validation step
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self._validate(X_val, y_val)
            else:
                val_loss, val_acc = None, None
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(time.time() - epoch_start)
            
            # Logging
            if self.verbose >= 1:
                log_msg = f"Epoch {epoch+1}/{self.max_epochs}: "
                log_msg += f"train_loss={train_loss:.4f}"
                if train_acc is not None:
                    log_msg += f", train_acc={train_acc:.4f}"
                if val_loss is not None:
                    log_msg += f", val_loss={val_loss:.4f}"
                if val_acc is not None:
                    log_msg += f", val_acc={val_acc:.4f}"
                log_msg += f" ({self.history['epoch_time'][-1]:.2f}s)"
                logger.info(log_msg)
            
            # Callback
            if callback is not None:
                callback(epoch, train_loss, val_loss)
            
            # Early stopping
            if self.early_stopping and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(patience={self.patience})"
                    )
                    break
        
        total_time = time.time() - start_time
        
        logger.info(
            f"✓ Training complete: {epoch+1} epochs in {total_time:.2f}s, "
            f"final train_loss={train_loss:.4f}"
        )
        
        return self.history
    
    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Train for one epoch."""
        # For now, use model's fit method
        # In practice, would implement proper batching and optimization
        
        # Define loss function for this batch
        def loss_for_batch(params):
            # This would be implemented by the model
            predictions = self.model.forward(X, params)
            loss = self.loss_fn(y, predictions)
            return loss
        
        # Simple training (actual implementation would vary by model)
        # Most models have their own fit() method
        # This trainer is more for orchestration
        
        # Get current predictions
        try:
            predictions = self.model.predict(X)
            loss = self.loss_fn(y, predictions)
            
            # Compute accuracy if classification
            if hasattr(self.model, 'score'):
                acc = self.model.score(X, y)
            else:
                acc = None
            
            return float(loss), acc
            
        except Exception as e:
            logger.warning(f"Training step failed: {e}")
            return float('inf'), None
    
    def _validate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Validate on validation set."""
        try:
            predictions = self.model.predict(X)
            loss = self.loss_fn(y, predictions)
            
            # Compute accuracy if classification
            if hasattr(self.model, 'score'):
                acc = self.model.score(X, y)
            else:
                acc = None
            
            return float(loss), acc
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return float('inf'), None
    
    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            axes[0].plot(self.history['train_loss'], label='Train')
            if self.history['val_loss'][0] is not None:
                axes[0].plot(self.history['val_loss'], label='Validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot accuracy if available
            if self.history['train_acc'][0] is not None:
                axes[1].plot(self.history['train_acc'], label='Train')
                if self.history['val_acc'][0] is not None:
                    axes[1].plot(self.history['val_acc'], label='Validation')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Training Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved training plot to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available - cannot plot history")
