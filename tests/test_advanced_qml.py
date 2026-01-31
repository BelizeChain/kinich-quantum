"""
Test Suite for Phase 3 Advanced QML Components.

Tests QSVM, Variational QNN, advanced feature maps, and training infrastructure.

Author: Kinich Quantum Team
License: MIT
"""

import pytest
import numpy as np


class TestQSVM:
    """Test Quantum Support Vector Machine."""
    
    def test_qsvm_initialization(self):
        """Test QSVM initialization."""
        from kinich.qml.classifiers.qsvm import QSVM
        
        qsvm = QSVM(num_features=4, kernel="ZZ", reps=2, C=1.0)
        
        assert qsvm.num_features == 4
        assert qsvm.kernel_type == "ZZ"
        assert qsvm.reps == 2
        assert qsvm.C == 1.0
    
    def test_quantum_kernel(self):
        """Test quantum kernel computation."""
        from kinich.qml.classifiers.qsvm import QuantumKernel
        
        kernel = QuantumKernel(num_features=3, feature_map="ZZ", reps=1)
        
        # Test single kernel evaluation
        x1 = np.array([0.5, 0.3, 0.2])
        x2 = np.array([0.4, 0.4, 0.1])
        
        k_val = kernel.evaluate(x1, x2)
        
        assert isinstance(k_val, float)
        assert 0 <= k_val <= 1  # Kernel should be normalized
    
    def test_qsvm_fit_predict(self):
        """Test QSVM training and prediction."""
        from kinich.qml.classifiers.qsvm import QSVM
        
        # Create simple dataset
        np.random.seed(42)
        X_train = np.random.randn(10, 4)
        y_train = np.random.randint(0, 2, 10)
        
        qsvm = QSVM(num_features=4, kernel="ZZ")
        
        # Fit
        qsvm.fit(X_train, y_train)
        
        # Predict
        X_test = np.random.randn(5, 4)
        predictions = qsvm.predict(X_test)
        
        assert predictions.shape == (5,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_qsvm_score(self):
        """Test QSVM scoring."""
        from kinich.qml.classifiers.qsvm import QSVM
        
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)
        
        qsvm = QSVM(num_features=3)
        qsvm.fit(X, y)
        
        accuracy = qsvm.score(X, y)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestVariationalQNN:
    """Test Variational Quantum Neural Networks."""
    
    def test_vqnn_initialization(self):
        """Test VQNN initialization."""
        from kinich.qml.models.variational_qnn import VariationalQNN
        
        vqnn = VariationalQNN(
            num_qubits=4,
            ansatz_type="hardware_efficient",
            reps=2,
            optimizer="SPSA"
        )
        
        assert vqnn.num_qubits == 4
        assert vqnn.ansatz_type == "hardware_efficient"
        assert vqnn.optimizer_type == "SPSA"
    
    def test_hardware_efficient_ansatz(self):
        """Test hardware-efficient ansatz."""
        from kinich.qml.models.variational_qnn import HardwareEfficientAnsatz
        
        ansatz = HardwareEfficientAnsatz(num_qubits=4, reps=3)
        
        num_params = ansatz.get_num_parameters()
        
        # Should be num_qubits * (reps + 1)
        expected = 4 * (3 + 1)
        assert num_params == expected
    
    def test_strongly_entangling_ansatz(self):
        """Test strongly-entangling ansatz."""
        from kinich.qml.models.variational_qnn import StronglyEntanglingAnsatz
        
        ansatz = StronglyEntanglingAnsatz(num_qubits=3, reps=2)
        
        num_params = ansatz.get_num_parameters()
        
        # Should be 3 * num_qubits * reps
        expected = 3 * 3 * 2
        assert num_params == expected
    
    def test_vqnn_forward(self):
        """Test VQNN forward pass."""
        from kinich.qml.models.variational_qnn import VariationalQNN
        
        vqnn = VariationalQNN(num_qubits=3, reps=1)
        
        X = np.random.randn(5, 3)
        output = vqnn.forward(X, vqnn.param_values)
        
        # Output should be [batch_size, 2^num_qubits]
        assert output.shape == (5, 8)
    
    def test_vqnn_training(self):
        """Test VQNN training."""
        from kinich.qml.models.variational_qnn import VariationalQNN
        
        np.random.seed(42)
        X_train = np.random.randn(10, 3)
        y_train = np.random.randint(0, 2, 10)
        
        vqnn = VariationalQNN(
            num_qubits=3,
            reps=1,
            optimizer="COBYLA",
            max_iter=10
        )
        
        # Train (limited iterations for speed)
        vqnn.fit(X_train, y_train)
        
        # Check history exists
        assert len(vqnn.history['loss']) > 0


class TestAdvancedFeatureMaps:
    """Test advanced feature maps."""
    
    def test_iqp_feature_map(self):
        """Test IQP feature map."""
        from kinich.qml.feature_maps.advanced_maps import IQPFeatureMap
        
        iqp = IQPFeatureMap(num_features=3, reps=2, entanglement="full")
        
        x = np.array([0.5, 0.3, 0.2])
        encoded = iqp.encode(x)
        
        # Should return probability distribution
        assert encoded.shape == (8,)  # 2^3
        assert np.isclose(np.sum(encoded), 1.0)  # Probabilities sum to 1
    
    def test_amplitude_encoding(self):
        """Test amplitude encoding."""
        from kinich.qml.feature_maps.advanced_maps import AmplitudeEncoding
        
        encoder = AmplitudeEncoding(num_qubits=2)
        
        x = np.array([1.0, 2.0, 3.0])  # Will be padded to 4
        encoded = encoder.encode(x)
        
        assert encoded.shape == (4,)  # 2^2
        assert np.isclose(np.linalg.norm(encoded), 1.0)  # Normalized
    
    def test_angle_encoding(self):
        """Test angle encoding."""
        from kinich.qml.feature_maps.advanced_maps import AngleEncoding
        
        encoder = AngleEncoding(num_features=3, rotation_axis="Y")
        
        x = np.array([0.5, 1.0, 1.5])
        encoded = encoder.encode(x)
        
        assert encoded.shape == (8,)  # 2^3
        assert np.isclose(np.sum(encoded), 1.0)
    
    def test_adaptive_feature_map(self):
        """Test adaptive feature map."""
        from kinich.qml.feature_maps.advanced_maps import AdaptiveFeatureMap
        
        adaptive = AdaptiveFeatureMap(num_features=3, base_map="IQP")
        
        x = np.array([0.5, 0.3, 0.2])
        encoded = adaptive.encode(x)
        
        assert encoded.shape == (8,)
        
        # Test parameter update
        grad_scale = np.random.randn(3)
        grad_shift = np.random.randn(3)
        adaptive.update_parameters(grad_scale, grad_shift)
        
        params = adaptive.get_parameters()
        assert 'scale' in params
        assert 'shift' in params


class TestOptimizers:
    """Test QML optimizers."""
    
    def test_spsa_optimizer(self):
        """Test SPSA optimizer."""
        from kinich.qml.training.optimizers import SPSAOptimizer
        
        # Simple quadratic loss
        def loss_fn(params):
            return np.sum(params ** 2)
        
        optimizer = SPSAOptimizer(max_iter=50, learning_rate=0.1)
        
        initial_params = np.array([1.0, 2.0, 3.0])
        result = optimizer.minimize(loss_fn, initial_params)
        
        assert 'params' in result
        assert 'loss' in result
        assert result['loss'] < loss_fn(initial_params)  # Should improve
    
    def test_cobyla_optimizer(self):
        """Test COBYLA optimizer."""
        from kinich.qml.training.optimizers import COBYLAOptimizer
        
        def loss_fn(params):
            return np.sum((params - 1) ** 2)
        
        optimizer = COBYLAOptimizer(max_iter=50)
        
        initial_params = np.zeros(3)
        result = optimizer.minimize(loss_fn, initial_params)
        
        assert result['loss'] < loss_fn(initial_params)
    
    def test_quantum_adam(self):
        """Test Quantum Adam optimizer."""
        from kinich.qml.training.optimizers import QuantumAdam
        
        def loss_fn(params):
            return np.sum(params ** 2)
        
        optimizer = QuantumAdam(learning_rate=0.1, max_iter=20)
        
        initial_params = np.array([2.0, 3.0])
        result = optimizer.minimize(loss_fn, initial_params)
        
        assert result['loss'] < loss_fn(initial_params)
        assert len(result['history']['gradients']) > 0


class TestLossFunctions:
    """Test QML loss functions."""
    
    def test_quantum_cross_entropy(self):
        """Test quantum cross-entropy loss."""
        from kinich.qml.training.losses import QuantumCrossEntropy
        
        loss_fn = QuantumCrossEntropy()
        
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        loss = loss_fn(y_true, y_pred)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_quantum_mse(self):
        """Test quantum MSE loss."""
        from kinich.qml.training.losses import QuantumMSE
        
        loss_fn = QuantumMSE()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        
        loss = loss_fn(y_true, y_pred)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_fidelity_loss(self):
        """Test fidelity loss."""
        from kinich.qml.training.losses import FidelityLoss
        
        loss_fn = FidelityLoss()
        
        # Identical states
        state1 = np.array([0.5, 0.3, 0.2])
        state1 = state1 / np.sum(state1)
        
        loss_same = loss_fn(state1, state1)
        assert np.isclose(loss_same, 0.0)  # Fidelity = 1, loss = 0
        
        # Different states
        state2 = np.array([0.2, 0.5, 0.3])
        state2 = state2 / np.sum(state2)
        
        loss_diff = loss_fn(state1, state2)
        assert loss_diff > 0


class TestQMLTrainer:
    """Test QML training infrastructure."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from kinich.qml.training.trainer import QMLTrainer
        from kinich.qml.models.variational_qnn import VariationalQNN
        
        model = VariationalQNN(num_qubits=3, max_iter=10)
        trainer = QMLTrainer(
            model=model,
            optimizer='SPSA',
            loss='mse',
            max_epochs=5
        )
        
        assert trainer.optimizer_type == 'SPSA'
        assert trainer.loss_type == 'mse'
        assert trainer.max_epochs == 5
    
    def test_trainer_training(self):
        """Test full training loop."""
        from kinich.qml.training.trainer import QMLTrainer
        from kinich.qml.models.variational_qnn import VariationalQNN
        
        np.random.seed(42)
        X_train = np.random.randn(20, 3)
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.randn(5, 3)
        y_val = np.random.randint(0, 2, 5)
        
        model = VariationalQNN(num_qubits=3, max_iter=5)
        model.fit(X_train, y_train)  # Pre-train slightly
        
        trainer = QMLTrainer(
            model=model,
            optimizer='SPSA',
            loss='mse',
            max_epochs=3,
            verbose=0
        )
        
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
