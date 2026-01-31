# 🚀 Kinich QML Deployment Guide

Production deployment of Kinich Quantum Machine Learning system on real quantum hardware.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Azure Quantum Setup](#azure-quantum-setup)
4. [IBM Quantum Setup](#ibm-quantum-setup)
5. [Configuration](#configuration)
6. [Deployment Patterns](#deployment-patterns)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Cost Optimization](#cost-optimization)
10. [Production Checklist](#production-checklist)

---

## Overview

This guide covers deploying Kinich QML to production quantum backends:
- **Azure Quantum**: Microsoft's cloud quantum platform (PRIMARY)
- **IBM Quantum**: IBM's quantum cloud services (FALLBACK)

**Recommendation**: Start with Azure Quantum for BelizeChain integration.

---

## Prerequisites

### Required Accounts

- [ ] Azure subscription with Quantum workspace
- [ ] IBM Quantum account (for fallback)
- [ ] Docker installed (for containerization)
- [ ] Kubernetes cluster (for orchestration)

### Required Python Packages

```bash
# Core dependencies
pip install azure-quantum qiskit qiskit-aer

# Deployment tools
pip install kubernetes docker-compose prometheus-client

# Monitoring
pip install opentelemetry-api opentelemetry-sdk
```

---

## Azure Quantum Setup

### Step 1: Create Azure Quantum Workspace

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create \
  --name belizechain-quantum \
  --location eastus

# Create Quantum workspace
az quantum workspace create \
  --resource-group belizechain-quantum \
  --name belizechain-qml \
  --location eastus \
  --storage-account belizechainqmlstorage
```

### Step 2: Add Quantum Providers

```bash
# Add IonQ provider (for trapped-ion hardware)
az quantum workspace provider add \
  --resource-group belizechain-quantum \
  --workspace-name belizechain-qml \
  --provider-id ionq \
  --provider-sku pay-as-you-go

# Add Quantinuum provider (for high-fidelity qubits)
az quantum workspace provider add \
  --resource-group belizechain-quantum \
  --workspace-name belizechain-qml \
  --provider-id quantinuum \
  --provider-sku pay-as-you-go
```

### Step 3: Configure Kinich Backend

```python
# kinich/config/production.py

from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

# Azure Quantum configuration
AZURE_QUANTUM_CONFIG = {
    'resource_group': 'belizechain-quantum',
    'workspace_name': 'belizechain-qml',
    'location': 'eastus',
    'subscription_id': 'YOUR_SUBSCRIPTION_ID'  # From Azure Portal
}

# Initialize workspace
workspace = Workspace(
    subscription_id=AZURE_QUANTUM_CONFIG['subscription_id'],
    resource_group=AZURE_QUANTUM_CONFIG['resource_group'],
    name=AZURE_QUANTUM_CONFIG['workspace_name'],
    location=AZURE_QUANTUM_CONFIG['location']
)

# Get provider
provider = AzureQuantumProvider(workspace=workspace)

# Select backend
backend = provider.get_backend('ionq.simulator')  # Start with simulator
# backend = provider.get_backend('ionq.qpu')  # Switch to real hardware
```

### Step 4: Update Kinich Configuration

```python
# kinich/core/quantum_node.py

from kinich.config.production import workspace, provider

class ProductionQuantumNode(QuantumNode):
    def __init__(self):
        super().__init__()
        self.workspace = workspace
        self.provider = provider
        self.backend = provider.get_backend('ionq.simulator')
    
    def submit_job(self, circuit):
        """Submit job to Azure Quantum"""
        job = self.backend.run(circuit, shots=1024)
        return job
    
    def get_result(self, job_id):
        """Retrieve job result"""
        job = self.backend.retrieve_job(job_id)
        result = job.result()
        return result.get_counts()
```

---

## IBM Quantum Setup

### Step 1: Get API Token

1. Go to [https://quantum-computing.ibm.com/](https://quantum-computing.ibm.com/)
2. Sign in and navigate to Account Settings
3. Copy your API token

### Step 2: Configure IBM Backend

```python
# kinich/config/production.py

from qiskit_ibm_runtime import QiskitRuntimeService

# IBM Quantum configuration
IBM_QUANTUM_CONFIG = {
    'token': 'YOUR_IBM_QUANTUM_TOKEN',
    'channel': 'ibm_quantum',
    'instance': 'ibm-q/open/main'  # Or your hub/group/project
}

# Save credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel=IBM_QUANTUM_CONFIG['channel'],
    token=IBM_QUANTUM_CONFIG['token'],
    instance=IBM_QUANTUM_CONFIG['instance'],
    overwrite=True
)

# Load service
service = QiskitRuntimeService()

# Get least busy backend
backend = service.least_busy(
    simulator=False,  # Real hardware
    operational=True,
    min_num_qubits=5
)
```

---

## Configuration

### Environment Variables

```bash
# .env.production

# Azure Quantum
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=belizechain-quantum
AZURE_QUANTUM_WORKSPACE=belizechain-qml
AZURE_QUANTUM_LOCATION=eastus

# IBM Quantum (Fallback)
IBM_QUANTUM_TOKEN=your-ibm-token
IBM_QUANTUM_INSTANCE=ibm-q/open/main

# Kinich Settings
KINICH_BACKEND=azure  # or 'ibm'
KINICH_USE_SIMULATOR=false  # true for testing
KINICH_SHOTS=1024
KINICH_MAX_RETRIES=3
KINICH_TIMEOUT=300

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Load Configuration

```python
# kinich/config/__init__.py

import os
from dotenv import load_dotenv

load_dotenv('.env.production')

class Config:
    # Azure
    AZURE_SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
    AZURE_RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
    AZURE_QUANTUM_WORKSPACE = os.getenv('AZURE_QUANTUM_WORKSPACE')
    
    # IBM
    IBM_QUANTUM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN')
    
    # Kinich
    BACKEND = os.getenv('KINICH_BACKEND', 'azure')
    USE_SIMULATOR = os.getenv('KINICH_USE_SIMULATOR', 'false') == 'true'
    SHOTS = int(os.getenv('KINICH_SHOTS', '1024'))
    MAX_RETRIES = int(os.getenv('KINICH_MAX_RETRIES', '3'))
    TIMEOUT = int(os.getenv('KINICH_TIMEOUT', '300'))
```

---

## Deployment Patterns

### Pattern 1: Standalone Quantum Service

```yaml
# docker-compose.yml

version: '3.8'

services:
  kinich-quantum:
    build: ./kinich
    environment:
      - KINICH_BACKEND=azure
      - KINICH_USE_SIMULATOR=false
    env_file:
      - .env.production
    ports:
      - "8000:8000"
    volumes:
      - ./kinich:/app
      - quantum-cache:/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  quantum-cache:
```

### Pattern 2: Kubernetes Deployment

```yaml
# k8s/kinich-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: kinich-quantum
  namespace: belizechain
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kinich-quantum
  template:
    metadata:
      labels:
        app: kinich-quantum
    spec:
      containers:
      - name: kinich
        image: belizechain/kinich:latest
        ports:
        - containerPort: 8000
        env:
        - name: KINICH_BACKEND
          value: "azure"
        - name: AZURE_SUBSCRIPTION_ID
          valueFrom:
            secretKeyRef:
              name: azure-quantum-secrets
              key: subscription-id
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: kinich-quantum-service
  namespace: belizechain
spec:
  selector:
    app: kinich-quantum
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Pattern 3: Hybrid Backend with Fallback

```python
# kinich/core/hybrid_backend.py

class HybridQuantumBackend:
    """Automatically switches between Azure and IBM"""
    
    def __init__(self):
        self.primary = AzureQuantumBackend()
        self.fallback = IBMQuantumBackend()
        self.current = self.primary
    
    def submit_job(self, circuit):
        """Submit with automatic fallback"""
        try:
            return self.primary.submit_job(circuit)
        except Exception as e:
            logger.warning(f"Azure failed: {e}, switching to IBM")
            self.current = self.fallback
            return self.fallback.submit_job(circuit)
    
    def get_backend_status(self):
        """Check which backend is active"""
        return {
            'active': 'azure' if self.current == self.primary else 'ibm',
            'primary_healthy': self.primary.health_check(),
            'fallback_healthy': self.fallback.health_check()
        }
```

---

## Performance Tuning

### 1. Circuit Optimization

```python
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

# Create optimization pipeline
pm = PassManager([
    Optimize1qGates(),
    CommutativeCancellation(),
])

# Apply before submission
optimized_circuit = pm.run(circuit)
```

### 2. Batch Job Submission

```python
# Submit multiple circuits in one batch
circuits = [circuit1, circuit2, circuit3]

jobs = backend.run(circuits, shots=1024)
results = [job.result() for job in jobs]
```

### 3. Caching Strategy

```python
import hashlib
import pickle

class QuantumCache:
    def __init__(self, cache_dir='/cache'):
        self.cache_dir = cache_dir
    
    def get_circuit_hash(self, circuit):
        """Generate unique hash for circuit"""
        circuit_str = circuit.qasm()
        return hashlib.sha256(circuit_str.encode()).hexdigest()
    
    def get_cached_result(self, circuit):
        """Retrieve cached result"""
        circuit_hash = self.get_circuit_hash(circuit)
        cache_file = f"{self.cache_dir}/{circuit_hash}.pkl"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_result(self, circuit, result):
        """Store result in cache"""
        circuit_hash = self.get_circuit_hash(circuit)
        cache_file = f"{self.cache_dir}/{circuit_hash}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

---

## Monitoring & Debugging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
quantum_jobs = Counter('quantum_jobs_total', 'Total quantum jobs submitted')
quantum_latency = Histogram('quantum_job_latency_seconds', 'Job execution time')
quantum_errors = Counter('quantum_errors_total', 'Total quantum errors')

# Instrument code
with quantum_latency.time():
    result = backend.run(circuit).result()
quantum_jobs.inc()
```

### Logging

```python
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/var/log/kinich/quantum.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('kinich.quantum')

# Log job details
logger.info(f"Submitting job: circuit_depth={circuit.depth()}, qubits={circuit.num_qubits}")
```

---

## Cost Optimization

### Azure Quantum Pricing (Example)

| Provider | Simulator | QPU (per shot) |
|----------|-----------|----------------|
| IonQ | Free | $0.00003 |
| Quantinuum | Free | $0.00030 |

### Cost-Saving Strategies

1. **Use Simulators for Development**
```python
if os.getenv('ENV') == 'development':
    backend = provider.get_backend('ionq.simulator')  # Free!
else:
    backend = provider.get_backend('ionq.qpu')
```

2. **Reduce Shot Count**
```python
# Development: Low shots
shots = 128 if os.getenv('ENV') == 'development' else 1024
```

3. **Batch Similar Jobs**
```python
# Group similar circuits together
batched_circuits = group_by_similarity(all_circuits)
for batch in batched_circuits:
    backend.run(batch, shots=1024)  # Single API call
```

4. **Cache Aggressively**
```python
cache = QuantumCache()

result = cache.get_cached_result(circuit)
if result is None:
    result = backend.run(circuit).result()
    cache.cache_result(circuit, result)
```

---

## Production Checklist

### Pre-Deployment

- [ ] Test on simulators thoroughly
- [ ] Run cost estimation for expected workload
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure logging aggregation
- [ ] Implement circuit caching
- [ ] Set resource limits (memory, CPU)
- [ ] Configure auto-scaling policies
- [ ] Set up alert rules
- [ ] Document runbooks for common issues

### Deployment

- [ ] Deploy to staging environment first
- [ ] Run smoke tests
- [ ] Gradual rollout (canary deployment)
- [ ] Monitor error rates
- [ ] Validate quantum results accuracy
- [ ] Check latency SLAs
- [ ] Verify fallback mechanisms
- [ ] Test disaster recovery

### Post-Deployment

- [ ] Monitor cost dashboards
- [ ] Review quantum job metrics
- [ ] Optimize slow circuits
- [ ] Tune error mitigation
- [ ] Update documentation
- [ ] Schedule regular audits
- [ ] Plan capacity increases

---

## Example: Complete Production Setup

```python
# production_kinich.py

import logging
from prometheus_client import start_http_server
from kinich.core.hybrid_backend import HybridQuantumBackend
from kinich.core.cache import QuantumCache
from kinich.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start metrics server
start_http_server(Config.PROMETHEUS_PORT)

# Initialize components
cache = QuantumCache(cache_dir='/var/cache/kinich')
backend = HybridQuantumBackend()

logger.info("Kinich Quantum Service Started")
logger.info(f"Backend: {Config.BACKEND}")
logger.info(f"Simulator Mode: {Config.USE_SIMULATOR}")

# Health check endpoint
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    status = backend.get_backend_status()
    return jsonify({
        'status': 'healthy',
        'backend': status
    })

@app.route('/ready')
def ready():
    return jsonify({'status': 'ready'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

**Previous**: [← API Reference](API_REFERENCE.md)  
**Next**: [Production Readiness →](../README.md)
