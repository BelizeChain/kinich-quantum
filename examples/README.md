# 🧪 Kinich QML Examples

Production-ready examples demonstrating quantum machine learning capabilities.

---

## 📚 Available Examples

### 1. QSVM vs Classical SVM (`01_qsvm_vs_classical_svm.py`)

**Description**: Compare Quantum Support Vector Machine (QSVM) against classical SVM on non-linear classification tasks.

**Runtime**: ~2-3 minutes  
**Level**: Beginner  

**What You'll Learn**:
- QSVM initialization and training
- Quantum kernel computation (ZZ kernel)
- Performance comparison with classical RBF kernel
- Decision boundary visualization
- Kernel matrix analysis

**Expected Results**:
- Classical SVM: ~85-88% accuracy
- Quantum SVM: ~87-90% accuracy (+2-5% improvement)

**Run**:
```bash
python 01_qsvm_vs_classical_svm.py
```

**Outputs**:
- `/tmp/qsvm_vs_classical.png` - Decision boundary comparison
- `/tmp/qsvm_confusion_matrices.png` - Confusion matrices
- `/tmp/qsvm_kernel_comparison.png` - Kernel matrix visualization

---

### 2. VQNN Training Loop (`02_vqnn_training_loop.py`)

**Description**: Train a Variational Quantum Neural Network (VQNN) from scratch with custom callbacks and validation monitoring.

**Runtime**: ~3-5 minutes  
**Level**: Intermediate  

**What You'll Learn**:
- VQNN configuration and initialization
- Custom training callbacks
- Validation monitoring
- Loss curve visualization
- Ansatz comparison (hardware-efficient vs strongly-entangling)

**Expected Results**:
- Training converges in ~50-100 iterations
- Final test accuracy: ~85-90%
- Clear learning curves

**Run**:
```bash
python 02_vqnn_training_loop.py
```

**Outputs**:
- `/tmp/vqnn_training_curves.png` - Training/validation loss and accuracy
- `/tmp/vqnn_confusion_matrix.png` - Test set confusion matrix
- `/tmp/vqnn_ansatz_comparison.png` - Ansatz performance comparison

---

### 3. Feature Map Comparison (`03_feature_map_comparison.py`)

**Description**: Benchmark all 6 feature encoding strategies across 4 different datasets to understand which encoding works best for different data patterns.

**Runtime**: ~5-7 minutes  
**Level**: Advanced  

**What You'll Learn**:
- All 6 feature maps (ZZ, Pauli, IQP, Amplitude, Angle, Adaptive)
- Feature map selection criteria
- Dataset-specific performance patterns
- Encoding characteristics and trade-offs

**Datasets Tested**:
1. Linear separable data
2. Two moons (non-linear)
3. Concentric circles (complex non-linear)
4. Multi-cluster data

**Feature Maps Tested**:
1. ZZ Feature Map (second-order Pauli)
2. Pauli Feature Map (first-order)
3. IQP Feature Map (polynomial)
4. Amplitude Encoding (direct)
5. Angle Encoding (rotation-based)
6. Adaptive Feature Map (learnable)

**Run**:
```bash
python 03_feature_map_comparison.py
```

**Outputs**:
- `/tmp/feature_map_heatmap.png` - Performance heatmap across datasets
- `/tmp/feature_map_bars.png` - Bar chart comparison per dataset

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn seaborn

# Install Kinich
pip install -e ../kinich
```

### Run All Examples

```bash
# Run examples sequentially
python 01_qsvm_vs_classical_svm.py
python 02_vqnn_training_loop.py
python 03_feature_map_comparison.py

# Or use script (if available)
./run_all_examples.sh
```

---

## 📊 Expected Outputs

### Performance Benchmarks

| Example | Classical Baseline | Quantum Result | Improvement |
|---------|-------------------|----------------|-------------|
| QSVM vs SVM | 85-88% | 87-90% | +2-5% |
| VQNN Training | N/A | 85-90% | N/A |
| Feature Maps | Varies | Dataset-dependent | Varies |

### Visualizations

Each example generates publication-quality plots saved to `/tmp/`:
- Decision boundaries
- Confusion matrices
- Training curves
- Performance comparisons
- Kernel visualizations
- Heatmaps

---

## 🎓 Learning Path

### Beginner Path
1. Start with **Example 1 (QSVM)** to understand quantum advantage
2. Learn about quantum kernels and their benefits
3. Compare with classical methods

### Intermediate Path
1. Move to **Example 2 (VQNN)** for training workflows
2. Understand callbacks and validation
3. Experiment with different ansatzes

### Advanced Path
1. Complete **Example 3 (Feature Maps)** for encoding strategies
2. Understand when to use each feature map
3. Adapt to your specific problem domain

---

## 🔧 Customization

### Modify Datasets

```python
# Example 1: Use your own dataset
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
# ... rest of code
```

### Change Hyperparameters

```python
# Example 2: Adjust training parameters
vqnn = VariationalQNN(
    num_qubits=8,        # Increase qubits
    num_layers=5,        # Deeper circuit
    optimizer="SPSA"     # Different optimizer
)
```

### Add Custom Metrics

```python
# Example 3: Track custom metrics
def custom_callback(iteration, loss):
    print(f"Custom metric: {compute_custom_metric()}")

trainer.train(X, y, callback=custom_callback)
```

---

## 🐛 Troubleshooting

### Example Fails to Run

**Problem**: Import errors
```
ModuleNotFoundError: No module named 'kinich'
```

**Solution**: Install Kinich
```bash
cd ../
pip install -e .
```

---

### Slow Execution

**Problem**: Examples take too long
```
Runtime > 10 minutes
```

**Solution**: Reduce iterations/samples
```python
# Reduce max_iter
vqc.fit(X, y, max_iter=50)  # Instead of 100

# Or reduce dataset size
X_small = X[:50]
```

---

### Missing Visualizations

**Problem**: Plots not saved
```
FileNotFoundError: /tmp/qsvm_vs_classical.png
```

**Solution**: Check write permissions
```bash
ls -la /tmp/
chmod 755 /tmp/
```

---

## 📚 Additional Resources

### Documentation
- [QML Architecture Guide](../docs/QML_ARCHITECTURE_GUIDE.md)
- [Hybrid Workflow Tutorial](../docs/HYBRID_WORKFLOW_TUTORIAL.md)
- [API Reference](../docs/API_REFERENCE.md)
- [Deployment Guide](../docs/DEPLOYMENT_GUIDE.md)

### Related Files
- Phase 1: [PHASE1_COMPLETE.md](../PHASE1_COMPLETE.md)
- Phase 2: [PHASE2_COMPLETE.md](../PHASE2_COMPLETE.md)
- Phase 3: [PHASE3_COMPLETE.md](../PHASE3_COMPLETE.md)
- Phase 4: [PHASE4_COMPLETE.md](../PHASE4_COMPLETE.md)

---

## 🎯 Next Steps

After completing these examples:

1. **Build Your Own Models**: Use examples as templates
2. **Try Real Quantum Hardware**: See [Deployment Guide](../docs/DEPLOYMENT_GUIDE.md)
3. **Integrate with Nawal**: Follow [Hybrid Tutorial](../docs/HYBRID_WORKFLOW_TUTORIAL.md)
4. **Optimize Performance**: Review [Architecture Guide](../docs/QML_ARCHITECTURE_GUIDE.md)

---

## 📝 Citation

If you use these examples in your research, please cite:

```bibtex
@software{belizechain_kinich_qml,
  title = {Kinich Quantum Machine Learning Examples},
  author = {BelizeChain Development Team},
  year = {2026},
  url = {https://github.com/belizechain/kinich}
}
```

---

**Questions?** See [API Reference](../docs/API_REFERENCE.md) or [Architecture Guide](../docs/QML_ARCHITECTURE_GUIDE.md)
