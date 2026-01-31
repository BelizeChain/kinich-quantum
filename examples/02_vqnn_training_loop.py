"""
Example 2: Variational Quantum Neural Network (VQNN) Training Loop

Demonstrates training a VQNN from scratch with:
- Custom training loop
- Validation monitoring
- Callback functions
- Loss visualization

Expected Results:
- Training converges in ~50-100 iterations
- Final accuracy: ~85-90%
- Clear learning curve

Runtime: ~3-5 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kinich.qml.models import VariationalQNN
from kinich.qml.training import QMLTrainer

# ============================================================
# 1. Prepare Dataset
# ============================================================

print("=" * 60)
print("Variational Quantum Neural Network Training")
print("=" * 60)

# Generate multi-class classification dataset
X, y = make_classification(
    n_samples=150,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\nDataset Statistics:")
print(f"  Training:   {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test:       {len(X_test)} samples")
print(f"  Features:   {X_train.shape[1]}")
print(f"  Classes:    {len(np.unique(y))}")

# ============================================================
# 2. Define VQNN Model
# ============================================================

print("\n" + "-" * 60)
print("Initializing Variational QNN...")
print("-" * 60)

vqnn = VariationalQNN(
    num_qubits=4,
    num_outputs=3,  # Multi-class classification
    num_layers=3,
    ansatz_type="hardware_efficient",  # Faster than strongly_entangling
    feature_map="ZZ",
    optimizer="COBYLA"
)

print(f"\n✅ Model Configuration:")
print(f"   Qubits:      {vqnn.num_qubits}")
print(f"   Outputs:     {vqnn.num_outputs}")
print(f"   Layers:      {vqnn.num_layers}")
print(f"   Ansatz:      {vqnn.ansatz_type}")
print(f"   Feature Map: {vqnn.feature_map}")
print(f"   Optimizer:   {vqnn.optimizer}")

# ============================================================
# 3. Training with Custom Callback
# ============================================================

print("\n" + "-" * 60)
print("Starting Training...")
print("-" * 60)

# Lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

def training_callback(iteration, loss, val_loss=None, val_acc=None):
    """Called after each training iteration"""
    train_losses.append(loss)
    
    if val_loss is not None:
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    # Print progress every 10 iterations
    if iteration % 10 == 0:
        status = f"Iter {iteration:3d} | Train Loss: {loss:.4f}"
        if val_loss is not None:
            status += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
        print(status)

# Create trainer
trainer = QMLTrainer(
    model=vqnn,
    optimizer="COBYLA",
    loss_function="cross_entropy",
    max_iter=100
)

# Train with validation monitoring
print("\nTraining Progress:")
print("-" * 60)

history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    callback=training_callback
)

print("-" * 60)
print("✅ Training Complete!")

# ============================================================
# 4. Evaluate on Test Set
# ============================================================

print("\n" + "-" * 60)
print("Test Set Evaluation...")
print("-" * 60)

test_pred = vqnn.predict(X_test)
test_acc = vqnn.score(X_test, y_test)

print(f"\n✅ Test Accuracy: {test_acc:.3f}")

# Per-class accuracy
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(
    y_test, test_pred,
    target_names=[f'Class {i}' for i in range(3)]
))

# ============================================================
# 5. Visualize Learning Curves
# ============================================================

print("\n" + "-" * 60)
print("Generating Visualizations...")
print("-" * 60)

# Loss curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
axes[0, 0].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Curve')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Validation loss
axes[0, 1].plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Both losses
axes[1, 0].plot(train_losses, label='Train', color='blue', linewidth=2)
axes[1, 0].plot(val_losses, label='Validation', color='orange', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Train vs Validation Loss')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Validation accuracy
axes[1, 1].plot(val_accuracies, label='Validation Accuracy', 
                color='green', linewidth=2)
axes[1, 1].axhline(y=test_acc, color='red', linestyle='--', 
                   label=f'Test Acc: {test_acc:.3f}')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Validation Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/tmp/vqnn_training_curves.png', dpi=150)
print("✅ Saved training curves to /tmp/vqnn_training_curves.png")

# ============================================================
# 6. Confusion Matrix
# ============================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(3)],
            yticklabels=[f'Class {i}' for i in range(3)])
plt.title(f'VQNN Confusion Matrix\nTest Accuracy: {test_acc:.3f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/tmp/vqnn_confusion_matrix.png', dpi=150)
print("✅ Saved confusion matrix to /tmp/vqnn_confusion_matrix.png")

# ============================================================
# 7. Compare Ansatz Types
# ============================================================

print("\n" + "-" * 60)
print("Comparing Different Ansatzes...")
print("-" * 60)

ansatz_results = {}

for ansatz_type in ["hardware_efficient", "strongly_entangling"]:
    print(f"\nTraining with {ansatz_type} ansatz...")
    
    model = VariationalQNN(
        num_qubits=4,
        num_outputs=3,
        num_layers=2,
        ansatz_type=ansatz_type,
        optimizer="COBYLA"
    )
    
    # Quick training (fewer iterations)
    model.fit(X_train, y_train, max_iter=50)
    
    acc = model.score(X_test, y_test)
    ansatz_results[ansatz_type] = acc
    print(f"   → {ansatz_type}: {acc:.3f}")

# Plot comparison
plt.figure(figsize=(8, 5))
plt.bar(ansatz_results.keys(), ansatz_results.values(), 
        color=['skyblue', 'lightcoral'])
plt.ylabel('Test Accuracy')
plt.title('VQNN Ansatz Comparison')
plt.ylim([0, 1])
for ansatz, acc in ansatz_results.items():
    plt.text(ansatz, acc + 0.02, f'{acc:.3f}', 
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/vqnn_ansatz_comparison.png', dpi=150)
print("\n✅ Saved ansatz comparison to /tmp/vqnn_ansatz_comparison.png")

# ============================================================
# 8. Summary
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"\nFinal Metrics:")
print(f"  Test Accuracy:        {test_acc:.3f}")
print(f"  Final Train Loss:     {train_losses[-1]:.4f}")
print(f"  Final Val Loss:       {val_losses[-1]:.4f}")
print(f"  Best Val Accuracy:    {max(val_accuracies):.3f}")
print(f"  Training Iterations:  {len(train_losses)}")

print(f"\nAnsatz Comparison:")
for ansatz, acc in ansatz_results.items():
    print(f"  {ansatz:25s}: {acc:.3f}")

print("\n" + "=" * 60)
print("✅ Example Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("1. VQNN supports multi-class classification out-of-the-box")
print("2. Validation monitoring helps detect overfitting")
print("3. Callbacks enable custom training logic")
print("4. Different ansatzes trade off expressivity vs circuit depth")
print("\nNext: Try example 03_feature_map_comparison.py")
