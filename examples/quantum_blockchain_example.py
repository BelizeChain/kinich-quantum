"""
Example: Submit Quantum Jobs to BelizeChain Blockchain

Demonstrates the updated quantum pallet integration using u8 index pattern
for Substrate v42 compatibility.

IMPORTANT: This example uses the new index-based API. The old enum-based API
is deprecated and will be removed in v2.0.
"""

import asyncio
import hashlib
import uuid
from pathlib import Path

# Import Kinich blockchain adapter
from kinich.blockchain.belizechain_adapter import BelizeChainAdapter
from kinich.blockchain.quantum_indices import (
    QuantumBackendIndex,
    JobStatusIndex,
    AchievementTypeIndex,
    get_index_summary,
)


async def example_submit_quantum_job():
    """Example: Submit a quantum job using backend index."""
    
    print("=" * 70)
    print("EXAMPLE 1: Submit Quantum Job with Backend Index")
    print("=" * 70)
    
    # Initialize adapter
    adapter = BelizeChainAdapter(
        node_url="ws://localhost:9944",
        keypair_seed=None  # Uses //Alice development key
    )
    
    if not await adapter.connect():
        print("❌ Failed to connect to BelizeChain")
        return
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create circuit (simple example)
        circuit_data = b"OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
        circuit_hash = hashlib.sha256(circuit_data).digest()
        
        print(f"\n📊 Job Details:")
        print(f"   Job ID: {job_id}")
        print(f"   Backend: Qiskit (index {QuantumBackendIndex.QISKIT})")
        print(f"   Qubits: 20")
        print(f"   Depth: 50")
        print(f"   Shots: 1000")
        
        # Submit job using backend_index parameter (NEW API)
        tx_hash = await adapter.submit_quantum_job(
            job_id=job_id,
            backend="qiskit",  # Still accepted for backwards compatibility
            backend_index=QuantumBackendIndex.QISKIT,  # ✅ PREFERRED
            circuit_hash=circuit_hash,
            num_qubits=20,
            circuit_depth=50,
            num_shots=1000
        )
        
        if tx_hash:
            print(f"\n✅ Job submitted successfully!")
            print(f"   Transaction: {tx_hash}")
            
            # Update status using status_index (NEW API)
            print(f"\n📝 Updating job status to Running...")
            status_tx = await adapter.update_job_status(
                job_id=job_id,
                status_index=JobStatusIndex.RUNNING  # ✅ PREFERRED
            )
            
            if status_tx:
                print(f"✅ Status updated: {status_tx}")
        else:
            print(f"\n❌ Job submission failed")
    
    finally:
        await adapter.disconnect()


async def example_mint_achievement_nft():
    """Example: Mint achievement NFT using achievement index."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Mint Achievement NFT with Achievement Index")
    print("=" * 70)
    
    adapter = BelizeChainAdapter()
    
    if not await adapter.connect():
        print("❌ Failed to connect to BelizeChain")
        return
    
    try:
        job_id = str(uuid.uuid4())
        
        print(f"\n🏆 NFT Details:")
        print(f"   Job ID: {job_id}")
        print(f"   Achievement: First Quantum Job (index {AchievementTypeIndex.FIRST_QUANTUM_JOB})")
        
        # Mint NFT using achievement_index parameter (NEW API)
        nft_id = await adapter.mint_achievement_nft(
            job_id=job_id,
            achievement_index=AchievementTypeIndex.FIRST_QUANTUM_JOB,  # ✅ PREFERRED
            metadata={
                "name": "First Quantum Job",
                "description": "Completed first quantum computation on BelizeChain",
                "image": "ipfs://QmFirstJob...",
                "attributes": [
                    {"trait_type": "Achievement", "value": "Pioneer"},
                    {"trait_type": "Qubits", "value": 20},
                ]
            },
            transferable=True
        )
        
        if nft_id:
            print(f"\n✅ Achievement NFT minted!")
            print(f"   NFT ID: #{nft_id}")
        else:
            print(f"\n❌ NFT minting failed")
    
    finally:
        await adapter.disconnect()


async def example_multiple_backends():
    """Example: Submit jobs to different quantum backends."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Backend Submissions")
    print("=" * 70)
    
    adapter = BelizeChainAdapter()
    
    if not await adapter.connect():
        print("❌ Failed to connect to BelizeChain")
        return
    
    try:
        # Circuit for testing
        circuit_data = b"OPENQASM 2.0; qreg q[5]; h q[0]; cx q[0],q[1];"
        circuit_hash = hashlib.sha256(circuit_data).digest()
        
        # Submit to multiple backends
        backends = [
            (QuantumBackendIndex.AZURE_IONQ, "Azure IonQ"),
            (QuantumBackendIndex.IBM_QUANTUM, "IBM Quantum"),
            (QuantumBackendIndex.QISKIT, "Qiskit Simulator"),
            (QuantumBackendIndex.SPINQ_GEMINI, "SpinQ Gemini"),
        ]
        
        print(f"\n📊 Submitting jobs to {len(backends)} backends:\n")
        
        for backend_index, backend_name in backends:
            job_id = str(uuid.uuid4())
            
            print(f"   Backend: {backend_name} (index {backend_index})")
            
            tx_hash = await adapter.submit_quantum_job(
                job_id=job_id,
                backend="",  # Empty string (not used)
                backend_index=backend_index,  # ✅ Use index directly
                circuit_hash=circuit_hash,
                num_qubits=5,
                circuit_depth=10,
                num_shots=100
            )
            
            if tx_hash:
                print(f"      ✅ Submitted: {tx_hash[:16]}...")
            else:
                print(f"      ❌ Failed")
            
            # Small delay between submissions
            await asyncio.sleep(0.5)
    
    finally:
        await adapter.disconnect()


async def example_query_job_status():
    """Example: Query job status from blockchain."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Query Job Status")
    print("=" * 70)
    
    adapter = BelizeChainAdapter()
    
    if not await adapter.connect():
        print("❌ Failed to connect to BelizeChain")
        return
    
    try:
        # Submit a job first
        job_id = str(uuid.uuid4())
        circuit_data = b"OPENQASM 2.0; qreg q[2]; h q[0];"
        circuit_hash = hashlib.sha256(circuit_data).digest()
        
        print(f"\n📝 Creating test job...")
        
        await adapter.submit_quantum_job(
            job_id=job_id,
            backend="",
            backend_index=QuantumBackendIndex.QISKIT,
            circuit_hash=circuit_hash,
            num_qubits=2,
            circuit_depth=1,
            num_shots=100
        )
        
        # Query job status
        print(f"\n🔍 Querying job status...")
        
        job = await adapter.get_job_status(job_id)
        
        if job:
            print(f"\n✅ Job found:")
            print(f"   Job ID: {job.job_id}")
            print(f"   Submitter: {job.submitter}")
            print(f"   Backend: {QuantumBackendIndex.to_name(job.backend.value)}")
            print(f"   Status: {JobStatusIndex.to_name(job.status.value)}")
            print(f"   Qubits: {job.num_qubits}")
            print(f"   Shots: {job.num_shots}")
            print(f"   DALLA Cost: {job.dalla_cost / 1e12:.6f} DALLA")
        else:
            print(f"\n❌ Job not found")
    
    finally:
        await adapter.disconnect()


async def example_index_validation():
    """Example: Validate indices before submission."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Index Validation")
    print("=" * 70)
    
    print("\n📋 Testing index validation:\n")
    
    # Valid indices
    test_cases = [
        (QuantumBackendIndex.validate(4), 4, "Backend: Qiskit"),
        (QuantumBackendIndex.validate(10), 10, "Backend: Invalid (10)"),
        (JobStatusIndex.validate(2), 2, "Status: Completed"),
        (JobStatusIndex.validate(5), 5, "Status: Invalid (5)"),
        (AchievementTypeIndex.validate(11), 11, "Achievement: Custom"),
        (AchievementTypeIndex.validate(12), 12, "Achievement: Invalid (12)"),
    ]
    
    for is_valid, index, description in test_cases:
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   {description:35} → {status}")
    
    # String to index conversion
    print("\n🔄 String to Index Conversion:\n")
    
    backends = ["azure_ionq", "qiskit", "unknown_backend"]
    for backend_str in backends:
        index = QuantumBackendIndex.from_string(backend_str)
        name = QuantumBackendIndex.to_name(index)
        print(f"   '{backend_str:20}' → index {index} ({name})")


def print_index_reference():
    """Print complete index reference."""
    print("\n" + "=" * 70)
    print("INDEX REFERENCE")
    print("=" * 70)
    print(get_index_summary())


async def main():
    """Run all examples."""
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  BelizeChain Quantum Pallet - Index-Based API Examples".center(68) + "║")
    print("║" + "  (Substrate v42 Compatible)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Print index reference first
    print_index_reference()
    
    # Run examples
    try:
        await example_submit_quantum_job()
        await example_mint_achievement_nft()
        await example_multiple_backends()
        await example_query_job_status()
        await example_index_validation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\n📚 Next Steps:")
    print("   1. Review docs/QUANTUM_PALLET_DECODING_RESOLUTION.md")
    print("   2. Update your code to use *_index parameters")
    print("   3. Test with live BelizeChain node at ws://localhost:9944")
    print("   4. Migrate from deprecated enum-based API by v2.0")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
