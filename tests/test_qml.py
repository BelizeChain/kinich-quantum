"""
Kinich QML Tests

Test suite for Quantum Machine Learning components.
"""

import pytest
import numpy as np

# Import Kinich QML components
from kinich.qml.models.qnn import QuantumNeuralNetwork
from kinich.qml.classifiers.vqc import VariationalQuantumClassifier
from kinich.qml.feature_maps.zz_feature_map import ZZFeatureMap
from kinich.qml.hybrid.nawal_bridge import NawalQuantumBridge


class TestQuantumNeuralNetwork:
    """Test Quantum Neural Network base class."""
    
    def test_initialization(self):
        """Test QNN initialization via concrete implementation."""
        from kinich.qml.models.circuit_qnn import CircuitQuantumNeuralNetwork
        
        qnn = CircuitQuantumNeuralNetwork(num_qubits=4, num_layers=2)
        
        assert qnn.num_qubits == 4
        assert qnn.num_layers == 2
    
    def test_parameter_initialization(self):
        """Test parameter initialization via concrete implementation."""
        from kinich.qml.models.circuit_qnn import CircuitQuantumNeuralNetwork
        
        qnn = CircuitQuantumNeuralNetwork(num_qubits=4, num_layers=2)
        
        # Should have initialized param_values
        assert qnn.param_values is not None
        assert len(qnn.param_values) > 0


class TestVariationalQuantumClassifier:
    """Test Variational Quantum Classifier."""
    
    def test_initialization(self):
        """Test VQC initialization."""
        vqc = VariationalQuantumClassifier(
            num_features=4,
            num_classes=2,
            ansatz_layers=2
        )
        
        assert vqc.num_features == 4
        assert vqc.num_classes == 2
        assert vqc.ansatz_layers == 2
        assert not vqc.is_fitted
    
    def test_fit_predict(self):
        """Test training and prediction."""
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 20)
        
        X_test = np.random.randn(5, 4)
        
        vqc = VariationalQuantumClassifier(
            num_features=4,
            num_classes=2,
            max_iter=10  # Quick test
        )
        
        # Train
        vqc.fit(X_train, y_train)
        assert vqc.is_fitted
        
        # Predict
        predictions = vqc.predict(X_test)
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 20)
        X_test = np.random.randn(5, 4)
        
        vqc = VariationalQuantumClassifier(
            num_features=4,
            num_classes=2,
            max_iter=5
        )
        
        vqc.fit(X_train, y_train)
        probas = vqc.predict_proba(X_test)
        
        assert probas.shape == (5, 2)
        # Probabilities sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


class TestZZFeatureMap:
    """Test ZZ Feature Map."""
    
    def test_initialization(self):
        """Test feature map initialization."""
        fm = ZZFeatureMap(num_features=4, reps=2)
        
        assert fm.num_features == 4
        assert fm.reps == 2
    
    def test_encode(self):
        """Test feature encoding."""
        fm = ZZFeatureMap(num_features=4, reps=2)
        
        # Single sample
        data = np.array([0.5, 0.3, 0.8, 0.1])
        encoded = fm.encode(data)
        
        assert encoded.shape == (1, 4)
        # Should be normalized to [0, 2π]
        assert np.all(encoded >= 0)
        assert np.all(encoded <= 2 * np.pi)
    
    def test_batch_encode(self):
        """Test batch encoding."""
        fm = ZZFeatureMap(num_features=4, reps=2)
        
        # Batch
        data = np.random.randn(10, 4)
        encoded = fm.encode(data)
        
        assert encoded.shape == (10, 4)
    
    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        fm_linear = ZZFeatureMap(num_features=4, entanglement='linear')
        fm_full = ZZFeatureMap(num_features=4, entanglement='full')
        fm_circular = ZZFeatureMap(num_features=4, entanglement='circular')
        
        assert len(fm_linear.get_entanglement_pattern()) == 3  # (0,1), (1,2), (2,3)
        assert len(fm_full.get_entanglement_pattern()) == 6  # All pairs
        assert len(fm_circular.get_entanglement_pattern()) == 4  # Linear + wrap


class TestNawalQuantumBridge:
    """Test Nawal-Kinich quantum bridge."""
    
    def test_initialization(self):
        """Test bridge initialization."""
        bridge = NawalQuantumBridge(
            classical_dim=128,
            quantum_dim=8
        )
        
        assert bridge.classical_dim == 128
        assert bridge.quantum_dim == 8
        assert bridge.encoding_matrix is not None
        assert bridge.decoding_matrix is not None
    
    def test_classical_to_quantum(self):
        """Test classical → quantum encoding."""
        bridge = NawalQuantumBridge(
            classical_dim=128,
            quantum_dim=8
        )
        
        # Classical features from Nawal
        classical_features = np.random.randn(5, 128)
        
        # Encode to quantum
        quantum_features = bridge.classical_to_quantum(classical_features)
        
        assert quantum_features.shape == (5, 8)
        # Should be normalized
        assert np.all(quantum_features >= 0)
        assert np.all(quantum_features <= 1)
    
    def test_quantum_to_classical(self):
        """Test quantum → classical decoding."""
        bridge = NawalQuantumBridge(
            classical_dim=128,
            quantum_dim=8
        )
        
        # Quantum results from Kinich
        quantum_results = np.random.rand(5, 8)
        
        # Decode back to classical
        classical_results = bridge.quantum_to_classical(quantum_results)
        
        assert classical_results.shape == (5, 128)
    
    def test_roundtrip(self):
        """Test classical → quantum → classical roundtrip."""
        bridge = NawalQuantumBridge(
            classical_dim=64,
            quantum_dim=8
        )
        
        # Original classical features
        original = np.random.randn(3, 64)
        
        # Encode → decode
        quantum = bridge.classical_to_quantum(original)
        reconstructed = bridge.quantum_to_classical(quantum)
        
        assert reconstructed.shape == original.shape
        # Won't be perfect due to compression (64 -> 8 -> 64)
        # Just verify shape preservation and no errors
        assert not np.any(np.isnan(reconstructed))
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        bridge = NawalQuantumBridge(
            classical_dim=128,
            quantum_dim=8
        )
        
        ratio = bridge.get_compression_ratio()
        assert ratio == 8 / 128  # 0.0625


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
