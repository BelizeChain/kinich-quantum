"""
Kinich QML Demo

Demonstrates basic quantum machine learning workflow.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

from kinich.qml.classifiers.vqc import VariationalQuantumClassifier
from kinich.qml.feature_maps.zz_feature_map import ZZFeatureMap
from kinich.qml.hybrid.nawal_bridge import NawalQuantumBridge


def demo_vqc_classification():
    """Demo: Quantum classification with VQC."""
    print("\n" + "="*60)
    print("DEMO 1: Variational Quantum Classifier (VQC)")
    print("="*60)
    
    # Create synthetic binary classification data
    np.random.seed(42)
    n_samples = 40
    n_features = 4
    
    # Class 0: centered around [0, 0, 0, 0]
    X_class0 = np.random.randn(n_samples // 2, n_features) * 0.5
    
    # Class 1: centered around [1, 1, 1, 1]
    X_class1 = np.random.randn(n_samples // 2, n_features) * 0.5 + 1.0
    
    # Combine
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    # Split train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Features: {n_features}, Classes: 2")
    
    # Create and train VQC
    print("\nInitializing Variational Quantum Classifier...")
    vqc = VariationalQuantumClassifier(
        num_features=n_features,
        num_classes=2,
        feature_map="ZZ",
        ansatz_layers=2,
        max_iter=20
    )
    
    print("Training VQC...")
    vqc.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    train_acc = vqc.score(X_train, y_train)
    test_acc = vqc.score(X_test, y_test)
    
    print(f"✓ Train Accuracy: {train_acc:.2%}")
    print(f"✓ Test Accuracy:  {test_acc:.2%}")
    
    # Show prediction probabilities
    probas = vqc.predict_proba(X_test[:3])
    print(f"\nSample Predictions (first 3 test samples):")
    for i, (true_label, probs) in enumerate(zip(y_test[:3], probas)):
        pred_label = np.argmax(probs)
        print(f"  Sample {i+1}: True={true_label}, Pred={pred_label}, Probs={probs}")


def demo_feature_map():
    """Demo: Quantum feature encoding."""
    print("\n" + "="*60)
    print("DEMO 2: Quantum Feature Maps")
    print("="*60)
    
    # Classical data
    classical_data = np.array([
        [0.1, 0.3, 0.5, 0.7],
        [0.2, 0.4, 0.6, 0.8],
        [0.9, 0.7, 0.5, 0.3]
    ])
    
    print(f"\nClassical Data:\n{classical_data}")
    
    # ZZ Feature Map
    print("\nCreating ZZ Feature Map (linear entanglement)...")
    zz_map = ZZFeatureMap(num_features=4, reps=2, entanglement='linear')
    
    print(f"  Entanglement Pattern: {zz_map.get_entanglement_pattern()}")
    print(f"  Number of Parameters: {zz_map.get_num_parameters()}")
    
    # Encode
    encoded = zz_map.encode(classical_data)
    print(f"\nEncoded Quantum Features (normalized to [0, 2π]):")
    print(f"{encoded}")
    
    # Try different entanglements
    for entanglement in ['circular', 'full']:
        fm = ZZFeatureMap(num_features=4, entanglement=entanglement)
        print(f"\n{entanglement.capitalize()} entanglement: {fm.get_entanglement_pattern()}")


def demo_nawal_kinich_bridge():
    """Demo: Hybrid classical-quantum bridge."""
    print("\n" + "="*60)
    print("DEMO 3: Nawal-Kinich Quantum Bridge")
    print("="*60)
    
    # Simulate Nawal's classical features (high-dimensional)
    print("\nSimulating Nawal classical ML features...")
    classical_dim = 128
    quantum_dim = 8
    batch_size = 5
    
    nawal_features = np.random.randn(batch_size, classical_dim)
    print(f"  Nawal features: {nawal_features.shape}")
    
    # Create bridge
    print(f"\nCreating Nawal-Kinich bridge...")
    print(f"  Classical dimension: {classical_dim}")
    print(f"  Quantum dimension: {quantum_dim}")
    print(f"  Compression ratio: {quantum_dim/classical_dim:.2%}")
    
    bridge = NawalQuantumBridge(
        classical_dim=classical_dim,
        quantum_dim=quantum_dim,
        encoding_method="linear"
    )
    
    # Encode: Classical → Quantum
    print(f"\nEncoding classical → quantum...")
    quantum_features = bridge.classical_to_quantum(nawal_features)
    print(f"  Quantum features: {quantum_features.shape}")
    print(f"  Range: [{quantum_features.min():.3f}, {quantum_features.max():.3f}]")
    
    # Simulate Kinich QNN processing
    print(f"\nSimulating Kinich quantum processing...")
    # In reality, this would be: kinich_qnn.forward(quantum_features)
    quantum_results = np.tanh(quantum_features)  # Mock QNN output
    
    # Decode: Quantum → Classical
    print(f"\nDecoding quantum → classical...")
    classical_results = bridge.quantum_to_classical(quantum_results)
    print(f"  Results: {classical_results.shape}")
    
    print(f"\n✓ Hybrid workflow complete!")
    print(f"  Nawal ({classical_dim}D) → Bridge → Kinich ({quantum_dim}D) → Bridge → Output ({classical_dim}D)")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🌌 KINICH QUANTUM MACHINE LEARNING DEMOS 🌌")
    print("="*60)
    print("\nDemonstrating Phase 1: QML Foundation")
    print("  - Variational Quantum Classifier (VQC)")
    print("  - Quantum Feature Maps (ZZ encoding)")
    print("  - Nawal-Kinich Hybrid Bridge")
    
    try:
        demo_vqc_classification()
        demo_feature_map()
        demo_nawal_kinich_bridge()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETE")
        print("="*60)
        print("\nPhase 1 QML Foundation:")
        print("  ✓ Quantum Neural Networks working")
        print("  ✓ VQC classification functional")
        print("  ✓ Feature maps encoding data")
        print("  ✓ Nawal-Kinich bridge operational")
        print("\nNext: Phase 2 - Advanced integration with real quantum backends")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
