# 🏗️ Kinich Quantum Machine Learning Architecture Guide

**Version**: 1.0  
**Date**: January 2026  
**Status**: Production Ready  

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Integration Patterns](#integration-patterns)
5. [Data Flow](#data-flow)
6. [Best Practices](#best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Kinich QML is a hybrid quantum-classical machine learning system designed for:
- **Quantum advantage**: Leveraging quantum feature spaces for non-linear problems
- **Production readiness**: Sklearn-compatible APIs, comprehensive testing
- **Flexibility**: Multiple algorithms, optimizers, and backends
- **Integration**: Seamless Nawal (classical ML) integration

### Key Features

✅ **3 Quantum ML Algorithms**:
- Variational Quantum Classifier (VQC)
- Quantum Support Vector Machine (QSVM)
- Variational Quantum Neural Network (VQNN)

✅ **4 Quantum Feature Maps**:
- ZZ Feature Map (entangling, Pauli-Z basis)
- Pauli Feature Map (X, Y, Z rotations)
- IQP (Instantaneous Quantum Polynomial)
- Amplitude/Angle/Adaptive Encoding

✅ **3 Quantum Optimizers**:
- COBYLA (gradient-free, noise-resistant)
- SPSA (Simultaneous Perturbation Stochastic Approximation)
- QuantumAdam (adaptive learning rates)

✅ **Hybrid Integration**:
- PyTorch autograd compatibility
- Async Nawal-Kinich connector
- Classical fallback mechanisms

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BelizeChain Platform                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │    Nawal     │◄───────►│    Kinich    │                  │
│  │ Classical ML │ Hybrid  │  Quantum ML  │                  │
│  └──────────────┘ Bridge  └──────────────┘                  │
│        │                         │                           │
│        │                         │                           │
│   ┌────▼────┐              ┌────▼─────┐                     │
│   │PyTorch  │              │ Qiskit   │                     │
│   │Transformers│            │Azure Q   │                     │
│   └─────────┘              └──────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: User Interface
- **Nawal Models**: Classical transformers, LLMs
- **Kinich API**: Quantum ML algorithms
- **Jupyter Notebooks**: Interactive experimentation

### Layer 2: Hybrid Integration
- **KinichQuantumConnector**: Async quantum processing
- **PyTorch Integration**: Autograd through quantum layers
- **HybridQuantumClassicalLLM**: Transformer + Quantum architecture

### Layer 3: Quantum ML Core
- **Classifiers**: VQC, QSVM
- **Models**: QNN, VariationalQNN
- **Feature Maps**: ZZ, Pauli, IQP, Advanced
- **Training**: Optimizers, loss functions, callbacks

### Layer 4: Quantum Backend
- **Simulators**: Qiskit Aer (development)
- **Real Hardware**: Azure Quantum, IBM Quantum
- **Execution**: Circuit optimization, error mitigation

---

## 🔧 Core Components

### 1. Quantum Neural Network (QNN)

**Base class** for all quantum models.

```python
from kinich.qml.models import QNN

qnn = QNN(
    num_qubits=4,
    num_layers=2,
    backend='simulator'  # or 'azure', 'ibm'
)

# Forward pass
output = qnn.forward(X, parameters)

# Backward pass (parameter shift rule)
gradients = qnn.backward(grad_output, X)
```

**Architecture**:
```
Input (classical) → Feature Map → Quantum Circuit → Measurement → Output
                     ↑
                Parameters (trainable)
```

**Key Methods**:
- `forward()`: Execute quantum circuit
- `backward()`: Compute gradients via parameter shift
- `get_num_parameters()`: Total trainable parameters
- `build_circuit()`: Construct quantum circuit

### 2. Variational Quantum Classifier (VQC)

**Sklearn-compatible** quantum classifier.

```python
from kinich.qml.classifiers import VQC

vqc = VQC(
    num_features=4,      # Number of input features
    num_classes=2,       # Binary or multi-class
    num_layers=3,        # Circuit depth
    optimizer='COBYLA'   # COBYLA, SPSA, Adam
)

# Fit (train)
vqc.fit(X_train, y_train)

# Predict
predictions = vqc.predict(X_test)

# Evaluate
accuracy = vqc.score(X_test, y_test)
```

**Use Cases**:
- Binary classification (2 classes)
- Multi-class classification (one-vs-rest)
- Non-linear decision boundaries
- Small to medium datasets (< 1000 samples)

### 3. Quantum Support Vector Machine (QSVM)

**Kernel-based** quantum classifier using quantum feature maps.

```python
from kinich.qml.classifiers import QSVM

qsvm = QSVM(
    num_features=2,
    kernel='ZZ',        # 'ZZ' or 'Pauli'
    reps=2,             # Feature map repetitions
    C=1.0               # SVM regularization
)

qsvm.fit(X_train, y_train)
predictions = qsvm.predict(X_test)
```

**Quantum Kernel**:
```
K(x, y) = |⟨φ(x)|φ(y)⟩|²

where φ is quantum feature map
```

**Advantages**:
- Exponential feature space (2^n dimensions)
- Non-linear kernels via entanglement
- Proven quantum advantage on certain datasets

### 4. Variational Quantum Neural Network (VQNN)

**Flexible QNN** with custom ansatzes and optimizers.

```python
from kinich.qml.models import VariationalQNN

vqnn = VariationalQNN(
    num_qubits=4,
    ansatz_type='hardware_efficient',  # or 'strongly_entangling'
    reps=3,
    optimizer='SPSA',
    max_iter=100
)

vqnn.fit(X_train, y_train)
```

**Ansatz Types**:

1. **Hardware-Efficient**:
   - Minimal gate count
   - Fast execution
   - Lower expressivity

2. **Strongly-Entangling**:
   - Maximum entanglement
   - Higher expressivity
   - More parameters

### 5. Feature Maps

Transform classical data into quantum states.

```python
from kinich.qml.feature_maps import (
    ZZFeatureMap,
    PauliFeatureMap,
    IQPFeatureMap,
    AmplitudeEncoding,
    AngleEncoding,
    AdaptiveFeatureMap
)

# ZZ Feature Map (default)
zz = ZZFeatureMap(num_features=4, reps=2)
encoded = zz.encode(x)

# IQP (polynomial interactions)
iqp = IQPFeatureMap(num_features=4, reps=2)
encoded = iqp.encode(x)

# Adaptive (learned parameters)
adaptive = AdaptiveFeatureMap(num_features=4)
adaptive.fit(X_train)  # Learn optimal encoding
encoded = adaptive.encode(x)
```

**Comparison**:

| Feature Map | Expressivity | Speed | Use Case |
|-------------|--------------|-------|----------|
| ZZ          | Medium       | Fast  | General classification |
| Pauli       | Medium       | Fast  | Multi-axis rotations |
| IQP         | High         | Medium| Non-linear problems |
| Amplitude   | Very High    | Slow  | Image/signal encoding |
| Angle       | Low          | Very Fast | Simple features |
| Adaptive    | Very High    | Medium| Data-dependent encoding |

### 6. Training Infrastructure

Unified training framework.

```python
from kinich.qml.training import QMLTrainer, SPSAOptimizer

# Create trainer
trainer = QMLTrainer(
    model=vqnn,
    optimizer=SPSAOptimizer(learning_rate=0.1),
    loss_function='cross_entropy'
)

# Train with validation
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    callbacks=[checkpoint_callback]
)

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.legend()
```

**Loss Functions**:
- `cross_entropy`: Classification
- `mse`: Regression
- `fidelity`: Quantum state fidelity

**Optimizers**:
- `COBYLA`: Gradient-free, robust
- `SPSA`: Quantum-native, noise-resistant
- `QuantumAdam`: Adaptive, momentum-based

---

## 🔗 Integration Patterns

### Pattern 1: Standalone Quantum Classifier

**Simple classification** with quantum advantage.

```python
from kinich.qml.classifiers import QSVM
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate XOR-like data
X, y = make_moons(n_samples=100, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train QSVM
qsvm = QSVM(num_features=2, kernel='ZZ')
qsvm.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {qsvm.score(X_test, y_test)}")
```

### Pattern 2: Hybrid Quantum-Classical Pipeline

**Combine classical preprocessing** with quantum classification.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from kinich.qml.classifiers import VQC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('vqc', VQC(num_features=4, num_layers=2))
])

# Train end-to-end
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Pattern 3: Nawal-Kinich Hybrid Model

**Enhance classical transformers** with quantum layers.

```python
from nawal.integration import KinichQuantumConnector
from nawal.models import HybridQuantumClassicalLLM
import torch

# Create hybrid model
model = HybridQuantumClassicalLLM(
    vocab_size=10000,
    hidden_dim=768,
    quantum_dim=8,
    num_layers=6,
    enable_quantum=True,
    quantum_position='middle'  # Insert quantum layer mid-network
)

# Forward pass
input_ids = torch.randint(0, 10000, (4, 128))
outputs = model(input_ids)
logits = outputs['logits']

# Quantum statistics
stats = outputs['quantum_stats']
print(f"Quantum calls: {stats['quantum_calls']}")
print(f"Cache hits: {stats['cache_hits']}")
```

### Pattern 4: PyTorch Integration

**Train quantum layers** with PyTorch autograd.

```python
import torch
import torch.nn as nn
from kinich.qml.hybrid import TorchQuantumNeuralNetwork

# Hybrid model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(64, 4)
        self.quantum = TorchQuantumNeuralNetwork(
            num_qubits=4,
            num_layers=2
        )
        self.decoder = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantum(x)  # Quantum layer
        x = self.decoder(x)
        return x

# Train normally
model = HybridModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()  # Gradients through quantum layer!
    optimizer.step()
```

---

## 📊 Data Flow

### Training Flow

```
1. Input Data (X, y)
        ↓
2. Preprocessing (scaling, encoding)
        ↓
3. Feature Map (classical → quantum)
        ↓
4. Quantum Circuit (parameterized gates)
        ↓
5. Measurement (quantum → classical probabilities)
        ↓
6. Loss Calculation
        ↓
7. Gradient Computation (parameter shift rule)
        ↓
8. Parameter Update (optimizer)
        ↓
   Repeat until convergence
```

### Inference Flow

```
1. Input Data (X)
        ↓
2. Preprocessing (same as training)
        ↓
3. Feature Map
        ↓
4. Quantum Circuit (trained parameters)
        ↓
5. Measurement
        ↓
6. Post-processing (argmax, softmax)
        ↓
7. Predictions (y_pred)
```

---

## ✅ Best Practices

### 1. Data Preparation

```python
# ✅ GOOD: Scale features to [0, 2π]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

# ❌ BAD: Large values cause phase wrapping issues
X_unscaled = X  # Values in [-1000, 1000]
```

### 2. Feature Map Selection

```python
# ✅ GOOD: Match map to problem structure
# For spatial data (images): Amplitude encoding
# For time series: Angle encoding
# For tabular: ZZ or IQP

# ❌ BAD: Always using same feature map
```

### 3. Hyperparameter Tuning

```python
# ✅ GOOD: Grid search on small subset
from sklearn.model_selection import GridSearchCV

params = {
    'num_layers': [2, 3, 4],
    'optimizer': ['COBYLA', 'SPSA'],
    'C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(QSVM(), params, cv=3)
grid.fit(X_train_small, y_train_small)

# ❌ BAD: Training full model for each combination
```

### 4. Circuit Depth

```python
# ✅ GOOD: Start shallow, increase if needed
vqc = VQC(num_layers=2)  # Try 2-4 layers first

# ❌ BAD: Deep circuits on noisy hardware
vqc = VQC(num_layers=20)  # Too deep for NISQ devices
```

### 5. Optimizer Selection

```python
# ✅ GOOD: Match optimizer to hardware
# Simulators: Adam (gradient-based)
# Real quantum: COBYLA or SPSA (gradient-free)

if backend == 'simulator':
    optimizer = 'Adam'
else:  # Real quantum hardware
    optimizer = 'SPSA'  # Noise-resistant

# ❌ BAD: Gradient-based on noisy hardware
```

### 6. Batch Processing

```python
# ✅ GOOD: Process in batches to avoid memory issues
batch_size = 32
for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    predictions = qsvm.predict(X_batch)

# ❌ BAD: Processing all at once
predictions = qsvm.predict(X)  # OOM for large X
```

---

## 🚀 Performance Optimization

### 1. Caching

```python
# Enable kernel caching for QSVM
qsvm = QSVM(num_features=4, kernel='ZZ')
qsvm.fit(X_train, y_train)  # Kernel matrix cached

# Reuse for multiple predictions (no recomputation)
pred1 = qsvm.predict(X_test1)
pred2 = qsvm.predict(X_test2)
```

### 2. Circuit Optimization

```python
# Use fewer repetitions for faster execution
fast_map = ZZFeatureMap(num_features=4, reps=1)  # Fast
slow_map = ZZFeatureMap(num_features=4, reps=5)  # Accurate but slow
```

### 3. Parallel Execution

```python
# Set shots for parallel measurements
qnn = QNN(num_qubits=4, shots=1024)  # More shots = higher accuracy
```

### 4. Early Stopping

```python
# Stop training when validation loss plateaus
from kinich.qml.training import EarlyStopping

callback = EarlyStopping(patience=10, min_delta=0.001)
trainer.train(X_train, y_train, X_val, y_val, callbacks=[callback])
```

---

## 🔍 Troubleshooting

### Issue 1: Low Accuracy

**Symptoms**: Model stuck at ~50% accuracy

**Solutions**:
```python
# 1. Try different feature map
qsvm = QSVM(kernel='Pauli')  # Instead of 'ZZ'

# 2. Increase circuit depth
vqc = VQC(num_layers=4)  # Instead of 2

# 3. More training iterations
vqnn = VariationalQNN(max_iter=200)  # Instead of 50

# 4. Check data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Issue 2: Slow Training

**Symptoms**: Training takes hours

**Solutions**:
```python
# 1. Reduce dataset size for hyperparameter search
X_small, _, y_small, _ = train_test_split(X, y, train_size=0.1)

# 2. Use gradient-free optimizer
vqnn = VariationalQNN(optimizer='COBYLA')  # Faster than SPSA

# 3. Reduce repetitions
feature_map = ZZFeatureMap(reps=1)  # Instead of 3
```

### Issue 3: Out of Memory

**Symptoms**: `MemoryError` during training

**Solutions**:
```python
# 1. Process in batches
for batch in batches(X, batch_size=32):
    model.partial_fit(batch, y_batch)

# 2. Reduce number of qubits
qnn = QNN(num_qubits=4)  # Instead of 10

# 3. Use sparse matrices (for simulators)
backend_options = {'method': 'statevector'}
```

### Issue 4: Gradient Vanishing

**Symptoms**: Gradients become very small

**Solutions**:
```python
# 1. Use parameter shift rule (built-in)
# Already handled by QNN.backward()

# 2. Initialize parameters carefully
params = np.random.randn(num_params) * 0.1  # Small initialization

# 3. Use adaptive optimizer
optimizer = QuantumAdam(learning_rate=0.01)
```

---

## 📚 Additional Resources

- **Qiskit Textbook**: [https://qiskit.org/textbook](https://qiskit.org/textbook)
- **Azure Quantum Docs**: [https://docs.microsoft.com/azure/quantum](https://docs.microsoft.com/azure/quantum)
- **Original Papers**:
  - VQC: Havlíček et al. (2019) - "Supervised learning with quantum-enhanced feature spaces"
  - QSVM: Schuld & Killoran (2019) - "Quantum machine learning in feature Hilbert spaces"
  - QNN: Farhi & Neven (2018) - "Classification with Quantum Neural Networks"

---

**Next**: [Hybrid Workflow Tutorial →](HYBRID_WORKFLOW_TUTORIAL.md)
