# Kinich Quantum Computing Orchestration

<div align="center">

![Kinich Logo](docs/images/kinich-logo.png)

**Hybrid quantum-classical computing infrastructure for distributed quantum workloads**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

</div>

---

## 📖 Overview

**Kinich** (named after the Mayan sun god) is a production-grade quantum computing orchestration platform designed for distributed quantum workloads. It provides a unified interface for executing quantum circuits across multiple quantum backends (Azure Quantum, IBM Quantum, Google Cirq, SpinQ) with built-in error mitigation, security, and sovereignty controls.

**Part of BelizeChain Ecosystem**: Kinich is a core component of BelizeChain's sovereign blockchain infrastructure, providing quantum computing capabilities for cryptography, optimization, simulation, and AI enhancement. While Kinich can run standalone for development/testing, it integrates tightly with other BelizeChain components in production:

- 🔗 **BelizeChain Blockchain**: Submit Proof of Quantum Work (PQW) for validator rewards
- 🤖 **Nawal AI**: Hybrid quantum-classical machine learning, PoUW rewards, SRS tracking  
- 📦 **Pakit Storage**: Store large quantum results with sovereign DAG storage proofs

See [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md) for complete integration details.

### Key Features

- 🌐 **Multi-Backend Support**: Seamlessly execute on Azure Quantum, IBM Quantum, Google Cirq, SpinQ, or local simulators
- ⚛️ **Error Mitigation**: Readout error correction, Zero-Noise Extrapolation (ZNE), symmetry verification
- 🔐 **Enterprise Security**: JWT/API key authentication, role-based access control, audit logging
- 🌍 **Data Sovereignty**: Enforce data residency rules, compliance tracking (GDPR, PIPEDA, Belize DPA)
- 📊 **Observability**: Distributed tracing, Prometheus metrics, SLA monitoring, alerting
- 🔗 **Blockchain Integration**: Proof of Quantum Work (PQW) for validator rewards
- 🚀 **High Performance**: Async I/O, job prioritization, circuit optimization
- 🛠️ **Developer Friendly**: Type-safe APIs, comprehensive documentation, examples

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install kinich-quantum

# Or install from source
git clone https://github.com/BelizeChain/kinich-quantum.git
cd kinich-quantum
pip install -e .

# With optional dependencies
pip install kinich-quantum[server,monitoring,dev]
```

### Basic Usage

```python
from kinich import QuantumNode, QuantumJob, JobType
from qiskit import QuantumCircuit

# Initialize quantum node
node = QuantumNode(node_id="my-node")

# Create a simple Bell state circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Submit job to Azure Quantum backend
job = QuantumJob(
    job_type=JobType.SIMULATION,
    circuit=circuit,
    backend="ionq.simulator",
    shots=1024
)

result = node.submit_job(job)
print(f"Results: {result.counts}")
```

### Running a Quantum Node

```bash
# Development mode (standalone, no BelizeChain integration)
export KINICH_ENABLE_BLOCKCHAIN_INTEGRATION=false
kinich-node --config config.yaml

# Production mode (full BelizeChain integration)
export KINICH_BLOCKCHAIN_WS_URL=ws://blockchain:9944
export KINICH_NAWAL_API_URL=http://nawal:8889
export KINICH_PAKIT_API_URL=http://pakit:8080
export KINICH_ENABLE_BLOCKCHAIN_INTEGRATION=true
kinich-node --config config-production.yaml
```

---

## 🏗️ Architecture

### Standalone Architecture (Development)

```
┌─────────────────────────────────────────────────────────────┐
│                     Kinich Quantum Node                     │
├─────────────────────────────────────────────────────────────┤
│  Job Submission API (REST/WebSocket)                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Security    │ Sovereignty  │  Error       │  Monitoring    │
│  Manager     │  Manager     │  Mitigation  │  Manager       │
├──────────────┴──────────────┴──────────────┴────────────────┤
│             Job Scheduler & Circuit Optimizer               │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Azure       │  IBM         │  Google      │  SpinQ         │
│  Adapter     │  Adapter     │  Adapter     │  Adapter       │
└──────────────┴──────────────┴──────────────┴────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼

### BelizeChain Integration Architecture (Production)

```
┌─────────────────────────────────────────────────────────────┐
│                    BelizeChain Ecosystem                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐      ┌──────────────┐                   │
│   │   Nawal AI   │◄────►│   Kinich     │                   │
│   │              │      │   Quantum    │                   │
│   │ • PoUW       │      │              │                   │
│   │ • SRS Track  │      │ • PQW Proofs │                   │
│   │ • Hybrid QML │      │ • Job Exec   │                   │
│   └──────┬───────┘      └──────┬───────┘                   │
│          │                     │                           │
│          │ ┌───────────────────┴──────────┐                │
│          ▼ ▼                              ▼                │
│   ┌─────────────────┐            ┌──────────────┐          │
│   │   Blockchain    │            │    Pakit     │          │
│   │                 │            │   Storage    │          │
│   │ • Quantum Pallet│◄───────────│              │          │
│   │ • Staking       │            │ • DAG Store  │          │
│   │ • Community     │            │ • Proofs     │          │
│   └─────────────────┘            └──────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
   Azure Quantum  IBM Quantum  Google Cirq    SpinQ Local
```

**See [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md) for detailed integration patterns.**

---

## 🔗 BelizeChain Integration

### Required Integration (Production)

Kinich integrates with three BelizeChain components:

1. **Blockchain (Substrate RPC)**  
   - Submit Proof of Quantum Work (PQW) to quantum pallet
   - Receive DALLA rewards for validated quantum computations
   - Track quantum achievements and milestones

2. **Nawal (HTTP REST)**  
   - Record quantum contributions for Social Responsibility Score (SRS)
   - Hybrid quantum-classical machine learning workflows
   - PoUW validation and reward distribution

3. **Pakit (HTTP REST)**  
   - Store large quantum results (> 1MB) in sovereign DAG storage
   - Register storage proofs on-chain via LandLedger pallet
   - Retrieve historical quantum results by Content ID (CID)

### Development Mode (Standalone)

For testing Kinich features without full BelizeChain stack:

```bash
# Start Kinich with integrations disabled
docker-compose up -d

# All BelizeChain integrations mocked
# Uses local qasm_simulator backend
# Results stored in local PostgreSQL
```

---
  Azure Quantum  IBM Quantum  Google Cirq   SpinQ Hardware
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **QuantumNode** | Main orchestration engine for job management |
| **JobScheduler** | Priority-based job queuing and execution |
| **CircuitOptimizer** | Quantum circuit optimization and transpilation |
| **ResultAggregator** | Combines results from multiple backends |
| **Backend Adapters** | Unified interface for quantum hardware/simulators |
| **SecurityManager** | Authentication, authorization, audit logging |
| **SovereigntyManager** | Data residency enforcement and compliance |
| **ErrorMitigator** | Quantum error correction and noise mitigation |
| **MonitoringManager** | Metrics, tracing, alerting, SLA tracking |

---

## 📚 Documentation

### Configuration

Create a `config.yaml` file:

```yaml
# Quantum backend configuration
quantum:
  max_qubits: 128
  max_circuit_depth: 1000
  default_backend: "azure.ionq.simulator"
  backends:
    - name: "azure.ionq.simulator"
      type: "azure"
      region: "us-west"
    - name: "ibm.simulator"
      type: "ibm"
      token: "${IBM_QUANTUM_TOKEN}"

# Security configuration
security:
  enable_authentication: true
  jwt_secret: "${JWT_SECRET}"
  session_timeout_minutes: 60
  max_failed_attempts: 5

# Data sovereignty
sovereignty:
  enforce_data_residency: true
  allowed_regions: ["belize", "us-west", "ca-central"]
  default_classification: "INTERNAL"

# Error mitigation
error_mitigation:
  enable_readout_mitigation: true
  enable_zero_noise_extrapolation: false  # Expensive
  calibration_interval_hours: 24

# Monitoring
monitoring:
  enable_distributed_tracing: true
  enable_prometheus_metrics: true
  alert_on_high_latency: true
  sla_target_ms: 5000
```

### Advanced Usage

#### Production Node with Full Security

```python
from kinich.core import ProductionQuantumNode, ProductionNodeConfig
from kinich.security import Role
from kinich.sovereignty import DataClassification

# Initialize production node
config = ProductionNodeConfig(
    enable_authentication=True,
    enforce_data_residency=True,
    enable_error_mitigation=True,
    config_file_path="config.yaml"
)

node = ProductionQuantumNode(config=config)

# Create user and authenticate
user = node.security_manager.create_user(
    username="researcher_alice",
    role=Role.SUBMITTER,
    blockchain_address="0xABC123..."
)

api_key = node.security_manager.generate_api_key(user.user_id)

# Submit job with sovereignty check
job_id = node.submit_job_authenticated(
    user_id=user.user_id,
    api_key=api_key,
    circuit=my_circuit,
    backend="azure.ionq.simulator",
    data_classification=DataClassification.SOVEREIGN
)

# Get error-mitigated results
results = node.get_job_result(job_id, mitigate_errors=True)
```

#### Custom Error Mitigation

```python
from kinich.error_mitigation import QuantumErrorMitigator, ErrorMitigationConfig

# Configure error mitigation
config = ErrorMitigationConfig(
    enable_readout_mitigation=True,
    enable_zero_noise_extrapolation=True,
    zne_scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
    enable_symmetry_verification=True,
    max_acceptable_error=0.1
)

mitigator = QuantumErrorMitigator(config)

# Calibrate for backend
error_matrix = mitigator.calibrate_readout_errors(
    backend_name="ionq.simulator",
    num_qubits=5,
    shots=10000
)

# Apply mitigation to raw counts
mitigated_counts = mitigator.mitigate_readout_errors(
    counts=raw_results.counts,
    backend_name="ionq.simulator",
    num_qubits=5
)
```

#### Blockchain Integration (Proof of Quantum Work)

```python
from kinich.blockchain import ConsensusConnector

# Connect to BelizeChain node
connector = ConsensusConnector(
    rpc_url="ws://validator.belizechain.org:9944"
)

# Submit quantum work proof
tx_hash = connector.submit_quantum_work(
    node_id="kinich-node-1",
    job_id=job_id,
    circuit_depth=100,
    num_qubits=20,
    execution_time_ms=5000,
    verification_hash=hash(result)
)

print(f"Proof submitted: {tx_hash}")
```

---

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/BelizeChain/kinich-quantum.git
cd kinich-quantum

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests (requires backends)
pytest -m integration

# Run with coverage
pytest --cov=kinich --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format kinich tests

# Lint code
ruff check kinich tests

# Type check
mypy kinich

# Run all checks
pre-commit run --all-files
```

---

## 🌐 Supported Quantum Backends

| Backend | Type | Status | Notes |
|---------|------|--------|-------|
| **Azure Quantum (IonQ)** | Simulator/Hardware | ✅ Production | Primary production backend |
| **Azure Quantum (Quantinuum)** | Hardware | ✅ Production | For high-fidelity circuits |
| **IBM Quantum** | Simulator/Hardware | ✅ Production | Fallback backend |
| **Google Cirq** | Simulator | ✅ Production | Development/testing |
| **SpinQ** | Hardware | 🧪 Beta | Community quantum computers |
| **Local Simulator** | Simulator | ✅ Production | Offline development |

---

## 🔐 Security & Compliance

Kinich implements comprehensive security and compliance features:

- **Authentication**: JWT tokens, API keys, multi-factor authentication (MFA)
- **Authorization**: Role-based access control (RBAC) with fine-grained permissions
- **Audit Logging**: Encrypted audit trails for all operations
- **Data Encryption**: TLS 1.3 for transport, AES-256 for storage
- **Compliance**: GDPR, PIPEDA, Belize Data Protection Act support
- **Sovereignty**: Enforce data residency rules by region/classification

---

## 📊 Monitoring & Observability

### Prometheus Metrics

```python
# Exposed metrics (http://localhost:9090/metrics)
kinich_jobs_submitted_total          # Total jobs submitted
kinich_jobs_completed_total          # Successfully completed jobs
kinich_jobs_failed_total             # Failed jobs
kinich_job_duration_seconds          # Job execution time histogram
kinich_active_jobs                   # Currently executing jobs
kinich_backend_utilization_percent   # Backend usage percentage
```

### Distributed Tracing

```python
from kinich.monitoring import MonitoringManager

monitoring = MonitoringManager()

# Start trace
span_id = monitoring.start_span('quantum_execution', attributes={
    'backend': 'ionq.simulator',
    'qubits': 20
})

# Add events
monitoring.add_span_event(span_id, 'circuit_optimized')
monitoring.add_span_event(span_id, 'submitted_to_backend')

# End span
monitoring.end_span(span_id, status='success')
```

---

## 🤝 Integration with BelizeChain

Kinich was designed for seamless integration with BelizeChain's blockchain infrastructure:

```python
# Economy Pallet: Quantum transaction optimization
from kinich.integration import optimize_transaction_routing

optimal_route = optimize_transaction_routing(
    source="Alice",
    destination="Bob",
    amount=1000,
    constraints={"minimize_fees": True}
)

# Governance Pallet: Quantum voting algorithms
from kinich.integration import quantum_vote_aggregation

result = quantum_vote_aggregation(
    votes=district_votes,
    algorithm="grover_search"
)

# Staking Pallet: Proof of Quantum Work validation
from kinich.blockchain import verify_quantum_work

is_valid = verify_quantum_work(
    proof=submitted_proof,
    difficulty=network_difficulty
)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Azure Quantum Team**: For excellent quantum cloud infrastructure
- **IBM Quantum**: For pioneering accessible quantum computing
- **Qiskit Community**: For robust quantum development tools
- **BelizeChain Team**: For sovereign blockchain vision

---

## 📬 Contact & Support

- **Website**: https://kinich.belizechain.org
- **Documentation**: https://docs.belizechain.org/kinich
- **GitHub Issues**: https://github.com/BelizeChain/kinich-quantum/issues
- **Discord**: https://discord.gg/belizechain
- **Email**: quantum@belizechain.org

---

<div align="center">

**Built with ❤️ by the BelizeChain Team**

*Empowering sovereign quantum computing for all*

</div>
