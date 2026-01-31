# 📖 Kinich QML API Reference

Complete reference for all Kinich Quantum Machine Learning APIs.

---

## 📑 Table of Contents

1. [Core Algorithms](#core-algorithms)
   - [QuantumNeuralNetwork](#quantumneuralnetwork)
   - [VQC (Variational Quantum Classifier)](#vqc)
   - [QSVM (Quantum Support Vector Machine)](#qsvm)
   - [VariationalQNN](#variationalqnn)

2. [Feature Maps](#feature-maps)
   - [ZZFeatureMap](#zzfeaturemap)
   - [PauliFeatureMap](#paulifeaturemap)
   - [IQPFeatureMap](#iqpfeaturemap)
   - [AmplitudeEncoding](#amplitudeencoding)
   - [AngleEncoding](#angleencoding)
   - [AdaptiveFeatureMap](#adaptivefeaturemap)

3. [Hybrid Integration](#hybrid-integration)
   - [TorchQuantumNeuralNetwork](#torchquantumneuralnetwork)
   - [HybridQuantumClassicalLLM](#hybridquantumclassicalllm)
   - [KinichQuantumConnector](#kinichquantumconnector)

4. [Training](#training)
   - [QMLTrainer](#qmltrainer)
   - [SPSAOptimizer](#spsaoptimizer)
   - [QuantumAdam](#quantumadam)
   - [Loss Functions](#loss-functions)

---

## Core Algorithms

### QuantumNeuralNetwork

Base class for all quantum neural network models.

```python
from kinich.qml.models import QuantumNeuralNetwork
```

#### Constructor

```python
QuantumNeuralNetwork(
    num_qubits: int,
    num_layers: int,
    feature_map: str = "ZZ",
    entanglement: str = "linear",
    backend: str = "statevector_simulator"
)
```

**Parameters**:
- `num_qubits` (int): Number of qubits (quantum features)
- `num_layers` (int): Number of variational layers
- `feature_map` (str): Feature encoding method. Options: `"ZZ"`, `"Pauli"`, `"IQP"`
- `entanglement` (str): Qubit connectivity. Options: `"linear"`, `"full"`, `"circular"`
- `backend` (str): Quantum simulator/hardware backend

**Returns**: `QuantumNeuralNetwork` instance

#### Methods

##### fit

```python
def fit(
    X: np.ndarray,
    y: np.ndarray,
    optimizer: str = "COBYLA",
    max_iter: int = 100,
    callback: Optional[Callable] = None
) -> 'QuantumNeuralNetwork'
```

Train the quantum neural network.

**Parameters**:
- `X` (ndarray): Training features, shape `(n_samples, n_features)`
- `y` (ndarray): Training labels, shape `(n_samples,)`
- `optimizer` (str): Optimization algorithm. Options: `"COBYLA"`, `"SPSA"`, `"Adam"`
- `max_iter` (int): Maximum optimization iterations
- `callback` (Callable, optional): Function called after each iteration

**Returns**: `self` (trained model)

**Example**:
```python
qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
qnn.fit(X_train, y_train, optimizer="COBYLA", max_iter=50)
```

##### predict

```python
def predict(X: np.ndarray) -> np.ndarray
```

Predict class labels for samples.

**Parameters**:
- `X` (ndarray): Input features, shape `(n_samples, n_features)`

**Returns**: `ndarray` - Predicted labels, shape `(n_samples,)`

**Example**:
```python
predictions = qnn.predict(X_test)
```

##### score

```python
def score(X: np.ndarray, y: np.ndarray) -> float
```

Calculate classification accuracy.

**Parameters**:
- `X` (ndarray): Test features
- `y` (ndarray): True labels

**Returns**: `float` - Accuracy score in [0, 1]

**Example**:
```python
accuracy = qnn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

---

### VQC

Variational Quantum Classifier for classification tasks.

```python
from kinich.qml.classifiers import VQC
```

#### Constructor

```python
VQC(
    num_qubits: int,
    num_classes: int = 2,
    num_layers: int = 2,
    ansatz: str = "RealAmplitudes",
    optimizer: str = "COBYLA",
    max_iter: int = 100
)
```

**Parameters**:
- `num_qubits` (int): Number of qubits
- `num_classes` (int): Number of output classes
- `num_layers` (int): Circuit depth
- `ansatz` (str): Variational form. Options: `"RealAmplitudes"`, `"EfficientSU2"`
- `optimizer` (str): Optimization method
- `max_iter` (int): Training iterations

**Example**:
```python
vqc = VQC(num_qubits=4, num_classes=3, num_layers=3)
vqc.fit(X_train, y_train)
predictions = vqc.predict(X_test)
```

---

### QSVM

Quantum Support Vector Machine using quantum kernels.

```python
from kinich.qml.classifiers import QSVM
```

#### Constructor

```python
QSVM(
    num_features: int,  # Note: NOT num_qubits!
    kernel_type: str = "ZZ",
    gamma: float = 1.0,
    C: float = 1.0
)
```

**Parameters**:
- `num_features` (int): Number of input features (**not** `num_qubits`)
- `kernel_type` (str): Quantum kernel. Options: `"ZZ"`, `"Pauli"`
- `gamma` (float): Kernel coefficient
- `C` (float): Regularization parameter

**Example**:
```python
qsvm = QSVM(num_features=4, kernel_type="ZZ", gamma=0.5)
qsvm.fit(X_train, y_train)
accuracy = qsvm.score(X_test, y_test)
```

#### Methods

##### get_kernel_matrix

```python
def get_kernel_matrix(X: np.ndarray) -> np.ndarray
```

Compute quantum kernel matrix.

**Parameters**:
- `X` (ndarray): Input data

**Returns**: `ndarray` - Kernel matrix, shape `(n_samples, n_samples)`

**Example**:
```python
K = qsvm.get_kernel_matrix(X_train)
print(f"Kernel matrix shape: {K.shape}")
```

---

### VariationalQNN

Advanced variational quantum neural network with multiple ansatzes.

```python
from kinich.qml.models import VariationalQNN
```

#### Constructor

```python
VariationalQNN(
    num_qubits: int,
    num_outputs: int,
    num_layers: int = 2,
    ansatz_type: str = "hardware_efficient",
    feature_map: str = "ZZ",
    optimizer: str = "COBYLA"
)
```

**Parameters**:
- `num_qubits` (int): Number of qubits
- `num_outputs` (int): Number of output neurons
- `num_layers` (int): Ansatz depth
- `ansatz_type` (str): Circuit structure. Options: `"hardware_efficient"`, `"strongly_entangling"`
- `feature_map` (str): Encoding strategy
- `optimizer` (str): Training algorithm

**Example**:
```python
vqnn = VariationalQNN(
    num_qubits=6,
    num_outputs=10,
    ansatz_type="strongly_entangling",
    num_layers=3
)
vqnn.fit(X_train, y_train, max_iter=200)
```

#### Methods

##### fit_with_validation

```python
def fit_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    **kwargs
) -> dict
```

Train with validation monitoring.

**Returns**: `dict` - Training history with keys `"train_loss"`, `"val_loss"`, `"val_accuracy"`

**Example**:
```python
history = vqnn.fit_with_validation(
    X_train, y_train,
    X_val, y_val,
    max_iter=100
)

import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
```

---

## Feature Maps

### ZZFeatureMap

Second-order Pauli-Z feature map with ZZ interactions.

```python
from kinich.qml.feature_maps import ZZFeatureMap
```

#### Constructor

```python
ZZFeatureMap(
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "linear"
)
```

**Parameters**:
- `num_qubits` (int): Number of qubits
- `reps` (int): Repetitions of encoding block
- `entanglement` (str): Connectivity pattern

**Example**:
```python
fmap = ZZFeatureMap(num_qubits=4, reps=2)
circuit = fmap.construct_circuit(x_input)
```

---

### IQPFeatureMap

Instantaneous Quantum Polynomial feature map.

```python
from kinich.qml.feature_maps import IQPFeatureMap
```

#### Constructor

```python
IQPFeatureMap(
    num_features: int,  # Number of input features
    reps: int = 3
)
```

**Parameters**:
- `num_features` (int): Input feature dimension
- `reps` (int): Circuit repetitions

**Example**:
```python
iqp = IQPFeatureMap(num_features=4, reps=3)
encoded = iqp.encode(X_train)
```

---

### AmplitudeEncoding

Directly encode classical data into quantum amplitudes.

```python
from kinich.qml.feature_maps import AmplitudeEncoding
```

#### Constructor

```python
AmplitudeEncoding(
    num_qubits: int  # Must satisfy 2^num_qubits >= num_features
)
```

**Parameters**:
- `num_qubits` (int): Number of qubits (determines max features: $2^n$)

**Example**:
```python
# Encode 16 features requires 4 qubits (2^4 = 16)
amp_enc = AmplitudeEncoding(num_qubits=4)
circuit = amp_enc.encode(features[:16])
```

---

### AdaptiveFeatureMap

Learnable feature map with trainable parameters.

```python
from kinich.qml.feature_maps import AdaptiveFeatureMap
```

#### Constructor

```python
AdaptiveFeatureMap(
    num_features: int,
    num_layers: int = 2,
    learning_rate: float = 0.01
)
```

**Parameters**:
- `num_features` (int): Input dimension
- `num_layers` (int): Encoding depth
- `learning_rate` (float): Parameter update rate

**Methods**:

##### update_parameters

```python
def update_parameters(gradient: np.ndarray) -> None
```

Update encoding parameters via gradient descent.

**Example**:
```python
adaptive_map = AdaptiveFeatureMap(num_features=4, num_layers=3)

for epoch in range(100):
    # Forward pass
    encoded = adaptive_map.encode(X_batch)
    loss = compute_loss(encoded, y_batch)
    
    # Backward pass
    grad = compute_gradient(loss)
    adaptive_map.update_parameters(grad)
```

---

## Hybrid Integration

### TorchQuantumNeuralNetwork

PyTorch-compatible quantum layer with autograd support.

```python
from kinich.qml.hybrid import TorchQuantumNeuralNetwork
```

#### Constructor

```python
TorchQuantumNeuralNetwork(
    num_qubits: int,
    num_layers: int,
    ansatz: str = "RealAmplitudes"
)
```

**Parameters**:
- `num_qubits` (int): Quantum dimension
- `num_layers` (int): Circuit depth
- `ansatz` (str): Variational form

**Usage in PyTorch Model**:
```python
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical1 = nn.Linear(100, 8)
        self.quantum = TorchQuantumNeuralNetwork(num_qubits=8, num_layers=2)
        self.classical2 = nn.Linear(256, 10)  # 2^8 = 256 quantum outputs
    
    def forward(self, x):
        x = self.classical1(x)
        x = self.quantum(x)  # Gradients flow through!
        x = self.classical2(x)
        return x

model = HybridModel()
optimizer = torch.optim.Adam(model.parameters())

# Train normally - quantum layer integrates seamlessly
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch['x'])
    loss = criterion(output, batch['y'])
    loss.backward()  # ← Gradients computed through quantum layer
    optimizer.step()
```

---

### HybridQuantumClassicalLLM

Full hybrid language model combining classical transformers with quantum enhancement.

```python
from nawal.models.hybrid_llm import HybridQuantumClassicalLLM
```

#### Constructor

```python
HybridQuantumClassicalLLM(
    vocab_size: int,
    hidden_dim: int = 768,
    quantum_dim: int = 8,
    num_layers: int = 12,
    quantum_position: str = "middle"
)
```

**Parameters**:
- `vocab_size` (int): Vocabulary size
- `hidden_dim` (int): Hidden dimension
- `quantum_dim` (int): Quantum layer dimension
- `num_layers` (int): Total transformer layers
- `quantum_position` (str): Where to insert quantum layer. Options: `"early"`, `"middle"`, `"late"`

**Example**:
```python
hybrid_llm = HybridQuantumClassicalLLM(
    vocab_size=50000,
    quantum_dim=4,
    quantum_position="middle"
)

# Standard training
optimizer = torch.optim.AdamW(hybrid_llm.parameters(), lr=1e-4)
for batch in train_loader:
    outputs = hybrid_llm(batch['input_ids'])
    loss = criterion(outputs, batch['labels'])
    loss.backward()
    optimizer.step()
```

---

### KinichQuantumConnector

Asynchronous quantum backend connector with caching.

```python
from nawal.integration import KinichQuantumConnector
```

#### Constructor

```python
KinichQuantumConnector(
    classical_dim: int,
    quantum_dim: int,
    model_type: str = "VQC",
    enable_cache: bool = True,
    cache_size: int = 1000
)
```

**Parameters**:
- `classical_dim` (int): Classical feature dimension
- `quantum_dim` (int): Quantum processing dimension
- `model_type` (str): Quantum model. Options: `"VQC"`, `"QSVM"`, `"QNN"`
- `enable_cache` (bool): Enable result caching
- `cache_size` (int): Maximum cached results

#### Methods

##### quantum_process

```python
async def quantum_process(features: np.ndarray) -> np.ndarray
```

Process features through quantum backend (async).

**Example**:
```python
import asyncio

connector = KinichQuantumConnector(
    classical_dim=768,
    quantum_dim=8,
    enable_cache=True
)

async def process_batch(features):
    result = await connector.quantum_process(features)
    return result

# Usage
quantum_output = asyncio.run(process_batch(classical_features))
```

##### get_statistics

```python
def get_statistics() -> dict
```

Get connector usage statistics.

**Returns**: `dict` with keys:
- `quantum_calls` (int): Total quantum calls
- `cache_hits` (int): Cache hits
- `cache_misses` (int): Cache misses
- `avg_latency` (float): Average processing time

**Example**:
```python
stats = connector.get_statistics()
print(f"Hit rate: {stats['cache_hits'] / stats['quantum_calls']:.2%}")
print(f"Avg latency: {stats['avg_latency']:.3f}s")
```

---

## Training

### QMLTrainer

Unified training framework for all QML models.

```python
from kinich.qml.training import QMLTrainer
```

#### Constructor

```python
QMLTrainer(
    model: Any,
    optimizer: str = "COBYLA",
    loss_function: str = "cross_entropy",
    max_iter: int = 100
)
```

**Parameters**:
- `model` (QNN/VQC/QSVM): Quantum model to train
- `optimizer` (str): Optimization algorithm
- `loss_function` (str): Loss metric. Options: `"cross_entropy"`, `"mse"`, `"fidelity"`
- `max_iter` (int): Maximum iterations

#### Methods

##### train

```python
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None
) -> dict
```

Execute training loop with optional validation.

**Returns**: `dict` - Training history

**Example**:
```python
from kinich.qml.training import QMLTrainer
from kinich.qml.models import VariationalQNN

model = VariationalQNN(num_qubits=4, num_outputs=2)
trainer = QMLTrainer(
    model=model,
    optimizer="SPSA",
    loss_function="cross_entropy",
    max_iter=200
)

history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    callback=lambda i, loss: print(f"Iter {i}: {loss:.4f}")
)

# Plot results
import matplotlib.pyplot as plt
plt.plot(history['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
```

---

### SPSAOptimizer

Simultaneous Perturbation Stochastic Approximation - quantum-native optimizer.

```python
from kinich.qml.training import SPSAOptimizer
```

#### Constructor

```python
SPSAOptimizer(
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    perturbation: float = 0.01
)
```

**Parameters**:
- `initial_params` (ndarray): Starting parameter values
- `learning_rate` (float): Step size
- `perturbation` (float): Finite difference epsilon

**Example**:
```python
optimizer = SPSAOptimizer(
    initial_params=np.random.randn(20),
    learning_rate=0.05
)

for iteration in range(100):
    params = optimizer.get_params()
    loss = objective_function(params)
    optimizer.step(loss)
```

---

### QuantumAdam

Adaptive learning rate optimizer for quantum circuits.

```python
from kinich.qml.training import QuantumAdam
```

#### Constructor

```python
QuantumAdam(
    initial_params: np.ndarray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999
)
```

**Parameters**:
- `initial_params` (ndarray): Initial parameters
- `learning_rate` (float): Step size
- `beta1` (float): First moment decay
- `beta2` (float): Second moment decay

**Example**:
```python
adam = QuantumAdam(
    initial_params=circuit.parameters,
    learning_rate=0.01
)

for epoch in range(100):
    loss, gradient = compute_loss_and_grad(adam.params)
    adam.update(gradient)
```

---

### Loss Functions

#### cross_entropy

```python
from kinich.qml.training.loss import cross_entropy

loss = cross_entropy(predictions, targets)
```

**Parameters**:
- `predictions` (ndarray): Model outputs, shape `(n_samples, n_classes)`
- `targets` (ndarray): True labels, shape `(n_samples,)`

**Returns**: `float` - Cross-entropy loss

---

#### mse

```python
from kinich.qml.training.loss import mse

loss = mse(predictions, targets)
```

Mean squared error loss.

---

#### fidelity_loss

```python
from kinich.qml.training.loss import fidelity_loss

loss = fidelity_loss(state1, state2)
```

Quantum state fidelity-based loss (1 - fidelity).

**Parameters**:
- `state1` (ndarray): Quantum state vector
- `state2` (ndarray): Target state vector

---

## 🔗 Quick Reference

| Component | Import Path | Primary Use Case |
|-----------|-------------|------------------|
| QNN | `kinich.qml.models.QuantumNeuralNetwork` | General quantum ML |
| VQC | `kinich.qml.classifiers.VQC` | Classification tasks |
| QSVM | `kinich.qml.classifiers.QSVM` | Kernel-based classification |
| VQNN | `kinich.qml.models.VariationalQNN` | Advanced architectures |
| TorchQNN | `kinich.qml.hybrid.TorchQuantumNeuralNetwork` | PyTorch integration |
| Connector | `nawal.integration.KinichQuantumConnector` | Async quantum processing |
| Trainer | `kinich.qml.training.QMLTrainer` | Training orchestration |

---

**Previous**: [← Hybrid Workflow Tutorial](HYBRID_WORKFLOW_TUTORIAL.md)  
**Next**: [Example Notebooks →](../examples/)
