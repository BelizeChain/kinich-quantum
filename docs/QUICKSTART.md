# Kinich-Quantum New Features - Quick Start Guide

**Version**: 2.0 (Post-Ecosystem Integration)  
**Date**: February 13, 2026  
**Audience**: Developers integrating with kinich-quantum

---

## 🚀 What's New in v2.0

This release adds **cross-chain bridging**, **zero-knowledge proofs**, and **enhanced consensus** to kinich-quantum. Here's how to use them.

---

## 1️⃣ DAG Storage with Content Addressing

### Basic Usage

```python
from kinich.storage import QuantumResultStore

# Initialize storage
store = QuantumResultStore(
    pakit_api_url="http://pakit.belizechain.org:8000",
    ipfs_api_url="/ip4/127.0.0.1/tcp/5001"
)

# Store quantum result (returns CID + proof)
cid, proof = await store.store_quantum_result(
    job_id="my-quantum-job",
    circuit_qasm=my_circuit,
    measurement_counts={"00": 512, "11": 512},
    backend="azure_ionq"
)

print(f"Stored at CID: {cid}")
print(f"Merkle root: {proof.root.hex()}")

# Retrieve later
result = await store.retrieve_quantum_result(cid)
print(result['counts'])  # {"00": 512, "11": 512}
```

### Advanced: Large Results with Chunking

```python
# Large results (>1MB) are automatically chunked
large_result = {
    "statevector": [complex(1.0, 0.0)] * 1_000_000,  # 10MB
    "metadata": {...}
}

cid, proof = await store.store_quantum_result(
    job_id="large-simulation",
    circuit_qasm=circuit,
    measurement_counts=large_result,
    backend="azure_ionq"
)
# Result is chunked into 256KB pieces, stored as DAG
# Retrieval automatically reconstructs from chunks

# Check storage availability
availability = await store.check_result_availability(cid)
print(f"Available on {len(availability['available_sources'])} sources")
```

---

## 2️⃣ Cross-Chain Bridging

### Ethereum Bridge

```python
from kinich.blockchain import (
    CrossChainBridge,
    EthereumBridge,
    BridgeDestination,
    ChainType,
    QuantumResultBridgeData
)

# Initialize Ethereum bridge
eth_bridge = EthereumBridge(
    rpc_url="https://sepolia.infura.io/v3/YOUR_KEY",
    chain_id=11155111,  # Sepolia testnet
    result_bridge_address="0x...",  # Your deployed contract
    private_key="0x..."  # Your signing key
)

# Prepare quantum result for bridging
result_data = QuantumResultBridgeData(
    job_id="quantum-job-123",
    circuit_hash=circuit_hash,
    result_cid="bafybeigdyrqwh...",
    result_hash=result_hash,
    shots=1024,
    backend="azure_ionq",
    execution_time_ms=500,
    fidelity=0.987
)

# Bridge to Ethereum
destination = BridgeDestination(
    chain_type=ChainType.ETHEREUM,
    account="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
)

bridge = CrossChainBridge(ethereum_bridge=eth_bridge)
tx = await bridge.bridge_quantum_result(
    job_id="quantum-job-123",
    destination=destination,
    result_data=result_data,
    generate_proof=True
)

print(f"Bridged to Ethereum: {tx.destination_tx_hash}")
```

### Polkadot XCM Bridge

```python
from kinich.blockchain import PolkadotXCMBridge, ParachainId

# Initialize XCM bridge
xcm_bridge = PolkadotXCMBridge(
    polkadot_url="wss://rpc.polkadot.io",
    kusama_url="wss://kusama-rpc.polkadot.io"
)

# Bridge to Acala parachain
destination = BridgeDestination(
    chain_type=ChainType.PARACHAIN,
    chain_id=ParachainId.ACALA,
    account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
)

tx = await xcm_bridge.bridge_quantum_result(
    result_data=result_data,
    destination=destination,
    proof=proof,
    sender_keypair=keypair
)

# Verify bridged result
confirmed, data = xcm_bridge.verify_bridged_result(
    tx.transaction_id,
    ChainType.PARACHAIN
)
print(f"XCM confirmed: {confirmed}")
```

### NFT Bridging

```python
from kinich.blockchain import NFTBridgeData

# Bridge achievement NFT to Ethereum
nft_data = NFTBridgeData(
    token_id="12345",
    achievement_type="grover_algorithm",
    metadata_cid="bafybeib...",
    original_owner="5GrwvaE...",
    minted_at=1234567890
)

tx = await bridge.bridge_achievement_nft(
    nft_id="12345",
    destination=BridgeDestination(ChainType.ETHEREUM, account="0x..."),
    nft_data=nft_data
)
# NFT now visible on Ethereum (OpenSea, etc.)
```

---

## 3️⃣ Zero-Knowledge Proofs

### Basic zkSNARK Proof

```python
from kinich.security import (
    ZKProofGenerator,
    ProofSystem,
    CircuitType
)

# Initialize ZK proof generator
zk_gen = ZKProofGenerator(
    default_proof_system=ProofSystem.ZKSNARK_GROTH16,
    enable_circuit_privacy=True,
    enable_result_privacy=True
)

# Generate proof (HIDES circuit and results)
proof = zk_gen.generate_circuit_proof(
    job_id="confidential-vqe",
    circuit_qasm=secret_circuit,      # PRIVATE - hidden from verifier
    measurement_counts=results,       # PRIVATE - hidden from verifier
    num_qubits=10,                    # PUBLIC
    num_gates=50,                     # PUBLIC
    backend="azure_ionq",             # PUBLIC
    circuit_type=CircuitType.VQE
)

print(f"Proof size: {proof.proof_size_bytes} bytes")  # ~200 bytes
print(f"Generation time: {proof.generation_time_ms:.2f}ms")

# Verify proof (without seeing private data!)
is_valid = zk_gen.verify_proof(proof)
print(f"Proof valid: {is_valid}")
```

### Batch Proofs for Efficiency

```python
# Generate proofs for 100 jobs at once (zkSTARK)
jobs = [
    {
        "job_id": f"job-{i}",
        "circuit_qasm": circuits[i],
        "measurement_counts": results[i],
        "num_qubits": 10,
        "num_gates": 50,
        "backend": "azure_ionq"
    }
    for i in range(100)
]

batch_proof = zk_gen.generate_batch_proof(
    jobs=jobs,
    proof_system=ProofSystem.ZKSTARK
)

print(f"Batch proof: {batch_proof.num_jobs} jobs")
print(f"Proof size: {batch_proof.proof_size_bytes} bytes")
print(f"Compression ratio: {batch_proof.compression_ratio():.2%}")
# Output: 50% compression - much smaller than 100 individual proofs!

# Verify batch proof
is_valid = zk_gen.verify_batch_proof(batch_proof)
```

### Privacy-Preserving Cross-Chain Bridge

```python
# Combine ZK proofs with cross-chain bridging
proof = zk_gen.generate_circuit_proof(...)

bridge_proof = BridgeProof(
    proof_type="zksnark_groth16",
    proof_data=proof.proof_data,
    zk_proof=proof  # Include ZK proof in bridge transaction
)

# Bridge to Ethereum with ZK proof
tx = await bridge.bridge_quantum_result(
    job_id="secret-job",
    destination=eth_destination,
    result_data=result_data,
    proof=bridge_proof  # Ethereum contract can verify proof!
)
# Circuit remains private, but Ethereum knows it's valid!
```

---

## 4️⃣ Enhanced Consensus Proofs

### Proof of Useful Work (PoUW)

```python
from kinich.blockchain import BelizeChainAdapter

adapter = BelizeChainAdapter()
await adapter.connect()

# Submit PoUW after quantum job completes
tx_hash = await adapter.submit_proof_of_useful_work(
    job_id="research-vqe",
    circuit_hash=circuit_hash,
    computation_complexity=1_000_000_000,  # 1B FLOPs
    energy_consumption_kwh=0.001,          # 1 Wh
    execution_time_ms=500,
    result_cid="bafybeigdyr...",
    usefulness_score=90  # High scientific value
)

print(f"PoUW submitted: {tx_hash}")
# Earns consensus rewards based on useful work!
```

### Proof of Storage Work (PoSW)

```python
# After storing result to IPFS/Pakit
tx_hash = await adapter.submit_proof_of_storage_work(
    result_cid="bafybeigdyrqwh...",
    merkle_root=merkle_tree.root,
    chunk_count=10,
    total_size_bytes=2_500_000,
    replication_factor=3,  # Stored on 3 nodes
    storage_nodes=["node1.belizechain.org", "node2", "node3"],
    challenge_responses=[response1, response2]  # Optional PoRep
)

print(f"PoSW submitted: {tx_hash}")
# Earns consensus rewards for providing storage!
```

### Proof of Data Work (PoDW)

```python
# After completing quantum job with error mitigation
tx_hash = await adapter.submit_proof_of_data_work(
    job_id="high-fidelity-job",
    result_cid="bafybeigdyr...",
    data_quality_score=95,
    fidelity_score=0.992,  # 99.2% fidelity
    error_mitigation_applied=True,
    calibration_data_cid="bafybeia5od...",
    peer_validations=4
)

print(f"PoDW submitted: {tx_hash}")
# Earns consensus rewards for data quality!
```

---

## 5️⃣ Complete Workflow Example

### End-to-End: Quantum Job → Storage → Bridge → Consensus

```python
from kinich.core import QuantumNode
from kinich.storage import QuantumResultStore
from kinich.blockchain import (
    BelizeChainAdapter,
    CrossChainBridge,
    EthereumBridge,
    BridgeDestination,
    ChainType
)
from kinich.security import ZKProofGenerator, ProofSystem

# 1. Execute quantum job
node = QuantumNode()
job = await node.submit_job(
    circuit_qasm=my_vqe_circuit,
    backend="azure_ionq",
    shots=1024
)
result = await node.get_result(job.job_id)

# 2. Generate ZK proof (hide proprietary circuit)
zk_gen = ZKProofGenerator(ProofSystem.ZKSNARK_GROTH16)
zk_proof = zk_gen.generate_circuit_proof(
    job_id=job.job_id,
    circuit_qasm=my_vqe_circuit,
    measurement_counts=result.counts,
    num_qubits=10,
    num_gates=50,
    backend="azure_ionq"
)

# 3. Store result with DAG addressing
store = QuantumResultStore()
cid, storage_proof = await store.store_quantum_result(
    job_id=job.job_id,
    circuit_qasm=my_vqe_circuit,
    measurement_counts=result.counts,
    backend="azure_ionq"
)

# 4. Submit consensus proofs to BelizeChain
adapter = BelizeChainAdapter()
await adapter.connect()

# PoUW: Prove useful work
await adapter.submit_proof_of_useful_work(
    job_id=job.job_id,
    circuit_hash=circuit_hash,
    computation_complexity=1_000_000_000,
    energy_consumption_kwh=0.001,
    execution_time_ms=result.execution_time_ms,
    result_cid=cid,
    usefulness_score=85
)

# PoSW: Prove storage
await adapter.submit_proof_of_storage_work(
    result_cid=cid,
    merkle_root=storage_proof.root,
    chunk_count=storage_proof.chunk_count,
    total_size_bytes=result.size_bytes,
    replication_factor=3,
    storage_nodes=["node1", "node2", "node3"]
)

# PoDW: Prove data quality
await adapter.submit_proof_of_data_work(
    job_id=job.job_id,
    result_cid=cid,
    data_quality_score=92,
    fidelity_score=0.987,
    error_mitigation_applied=True,
    peer_validations=2
)

# 5. Bridge result to Ethereum (with ZK proof)
eth_bridge = EthereumBridge(rpc_url="...", ...)
bridge = CrossChainBridge(ethereum_bridge=eth_bridge)

tx = await bridge.bridge_quantum_result(
    job_id=job.job_id,
    destination=BridgeDestination(ChainType.ETHEREUM, account="0x..."),
    result_data=result_data,
    zk_proof=zk_proof  # Include privacy-preserving proof
)

print(f"✅ Complete workflow finished!")
print(f"   - Result stored at CID: {cid}")
print(f"   - Consensus proofs submitted (PoUW, PoSW, PoDW)")
print(f"   - Bridged to Ethereum: {tx.destination_tx_hash}")
print(f"   - Circuit privacy preserved via zkSNARK")
```

---

## 📚 API Reference

### Storage Module
- `QuantumResultStore`: Main storage interface
  - `.store_quantum_result()`: Store with CID generation
  - `.retrieve_quantum_result()`: Retrieve by CID
  - `.check_result_availability()`: Health check
  - `.generate_replication_proof()`: PoRep challenges

### Blockchain Module
- `BelizeChainAdapter`: BelizeChain integration
  - `.submit_proof_of_useful_work()`: PoUW submission
  - `.submit_proof_of_storage_work()`: PoSW submission
  - `.submit_proof_of_data_work()`: PoDW submission
  
- `CrossChainBridge`: Multi-chain orchestrator
  - `.bridge_quantum_result()`: Bridge results
  - `.bridge_achievement_nft()`: Bridge NFTs
  - `.verify_bridged_result()`: Verify cross-chain

- `EthereumBridge`: Ethereum Web3 integration
  - `.bridge_quantum_result()`: Call smart contract
  - `.bridge_nft()`: Mint wrapped NFT

- `PolkadotXCMBridge`: Polkadot XCM integration
  - `.bridge_quantum_result()`: Send XCM message
  - `.bridge_nft()`: Asset teleportation

### Security Module
- `ZKProofGenerator`: Zero-knowledge proof system
  - `.generate_circuit_proof()`: Single job proof
  - `.generate_batch_proof()`: Batched proof
  - `.verify_proof()`: Verify zkSNARK/zkSTARK

---

## 🐛 Troubleshooting

### Issue: "substrate-interface not installed"
```bash
pip install substrate-interface
```

### Issue: "py-ecc not available"
```bash
pip install py-ecc eth-hash[pycryptodome]
```

### Issue: "IPFS connection failed"
```bash
# Start local IPFS daemon
ipfs daemon

# Or use Pakit API URL
store = QuantumResultStore(pakit_api_url="http://pakit:8000")
```

### Issue: "Ethereum transaction failed (gas)"
```python
# Increase gas limit
eth_bridge = EthereumBridge(
    ...,
    gas_limit=500_000  # Increase if needed
)
```

---

## 🎓 Best Practices

1. **Always use ZK proofs for proprietary circuits**
   - Protects intellectual property
   - Reduces on-chain verification cost

2. **Submit all three consensus proofs**
   - Maximizes consensus rewards
   - Ensures result legitimacy

3. **Store large results with chunking**
   - Better for retrieval
   - Enables partial downloads

4. **Use batch proofs for efficiency**
   - 50%+ size reduction
   - Lower verification costs

5. **Bridge results to relevant chains**
   - Ethereum: DeFi applications
   - Polkadot: Parachain integrations

---

## 📖 Further Reading

- [AUDIT_REPORT.md](AUDIT_REPORT.md) - Technical audit details
- [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md) - Feature documentation
- [FINAL_IMPLEMENTATION_SUMMARY.md](FINAL_IMPLEMENTATION_SUMMARY.md) - Executive summary

---

**Ready to build quantum dApps? Let's go! 🚀**
