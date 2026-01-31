"""
Quantum Pallet Index Mappings for BelizeChain

This module provides index constants for quantum pallet enums that use the u8 index
pattern required by Substrate v42. All blockchain interactions must use these numeric
indices instead of enum variants.

Reference: docs/QUANTUM_PALLET_DECODING_RESOLUTION.md
"""


class QuantumBackendIndex:
    """
    Quantum computing backend indices.
    
    Maps to QuantumBackend enum in pallet-belize-quantum.
    Total variants: 8 (indices 0-7)
    """
    AZURE_IONQ = 0
    AZURE_QUANTINUUM = 1
    AZURE_RIGETTI = 2
    IBM_QUANTUM = 3
    QISKIT = 4
    SPINQ_GEMINI = 5
    SPINQ_TRIANGULUM = 6
    OTHER = 7
    
    # String mapping for convenience
    _STRING_MAP = {
        'azure_ionq': AZURE_IONQ,
        'azure_quantinuum': AZURE_QUANTINUUM,
        'azure_rigetti': AZURE_RIGETTI,
        'ibm_quantum': IBM_QUANTUM,
        'qiskit': QISKIT,
        'spinq_gemini': SPINQ_GEMINI,
        'spinq_triangulum': SPINQ_TRIANGULUM,
        'other': OTHER,
    }
    
    _INDEX_TO_NAME = {
        AZURE_IONQ: 'AzureIonQ',
        AZURE_QUANTINUUM: 'AzureQuantinuum',
        AZURE_RIGETTI: 'AzureRigetti',
        IBM_QUANTUM: 'IBMQuantum',
        QISKIT: 'Qiskit',
        SPINQ_GEMINI: 'SpinQGemini',
        SPINQ_TRIANGULUM: 'SpinQTriangulum',
        OTHER: 'Other',
    }
    
    @classmethod
    def from_string(cls, backend_str: str) -> int:
        """
        Convert backend string to index.
        
        Args:
            backend_str: Backend name (case-insensitive)
        
        Returns:
            Backend index (0-7)
        """
        return cls._STRING_MAP.get(backend_str.lower(), cls.OTHER)
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """
        Convert index to backend name.
        
        Args:
            index: Backend index (0-7)
        
        Returns:
            Backend name string
        """
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate backend index is in range."""
        return 0 <= index <= 7


class JobStatusIndex:
    """
    Quantum job status indices.
    
    Maps to JobStatus enum in pallet-belize-quantum.
    Total variants: 5 (indices 0-4)
    """
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    
    _INDEX_TO_NAME = {
        PENDING: 'Pending',
        RUNNING: 'Running',
        COMPLETED: 'Completed',
        FAILED: 'Failed',
        CANCELLED: 'Cancelled',
    }
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """Convert index to status name."""
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate status index is in range."""
        return 0 <= index <= 4


class VerificationStatusIndex:
    """
    Result verification status indices.
    
    Maps to VerificationStatus enum in pallet-belize-quantum.
    Total variants: 4 (indices 0-3)
    """
    UNVERIFIED = 0
    VERIFYING = 1
    VERIFIED = 2
    FAILED = 3
    
    _INDEX_TO_NAME = {
        UNVERIFIED: 'Unverified',
        VERIFYING: 'Verifying',
        VERIFIED: 'Verified',
        FAILED: 'Failed',
    }
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """Convert index to verification status name."""
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate verification status index is in range."""
        return 0 <= index <= 3


class AchievementTypeIndex:
    """
    Quantum achievement type indices for NFT minting.
    
    Maps to AchievementType enum in pallet-belize-quantum.
    Total variants: 12 (indices 0-11)
    """
    FIRST_QUANTUM_JOB = 0
    GROVER_ALGORITHM = 1
    SHOR_ALGORITHM = 2
    QUANTUM_FOURIER_TRANSFORM = 3
    VQE_ALGORITHM = 4
    QAOA_ALGORITHM = 5
    ACCURACY_95 = 6
    ACCURACY_99 = 7
    VOLUME_CONTRIBUTOR_100 = 8
    VOLUME_CONTRIBUTOR_1000 = 9
    ERROR_MITIGATION_CHAMPION = 10
    CUSTOM = 11
    
    _INDEX_TO_NAME = {
        FIRST_QUANTUM_JOB: 'FirstQuantumJob',
        GROVER_ALGORITHM: 'GroverAlgorithm',
        SHOR_ALGORITHM: 'ShorAlgorithm',
        QUANTUM_FOURIER_TRANSFORM: 'QuantumFourierTransform',
        VQE_ALGORITHM: 'VQEAlgorithm',
        QAOA_ALGORITHM: 'QAOAAlgorithm',
        ACCURACY_95: 'Accuracy95',
        ACCURACY_99: 'Accuracy99',
        VOLUME_CONTRIBUTOR_100: 'VolumeContributor100',
        VOLUME_CONTRIBUTOR_1000: 'VolumeContributor1000',
        ERROR_MITIGATION_CHAMPION: 'ErrorMitigationChampion',
        CUSTOM: 'Custom',
    }
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """Convert index to achievement type name."""
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate achievement type index is in range."""
        return 0 <= index <= 11


class VerificationVoteIndex:
    """
    Verification vote indices.
    
    Maps to VerificationVote enum in pallet-belize-quantum.
    Total variants: 3 (indices 0-2)
    """
    APPROVE = 0
    REJECT = 1
    ABSTAIN = 2
    
    _INDEX_TO_NAME = {
        APPROVE: 'Approve',
        REJECT: 'Reject',
        ABSTAIN: 'Abstain',
    }
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """Convert index to vote name."""
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate vote index is in range."""
        return 0 <= index <= 2


class ChainDestinationIndex:
    """
    Cross-chain bridge destination indices.
    
    Maps to ChainDestination enum in pallet-belize-quantum.
    Total variants: 5 (indices 0-4)
    
    Note: Parachain variant contains parachain_id which must be handled separately.
    """
    ETHEREUM = 0
    POLKADOT = 1
    KUSAMA = 2
    PARACHAIN = 3  # Requires parachain_id parameter
    BELIZE_CHAIN = 4
    
    _INDEX_TO_NAME = {
        ETHEREUM: 'Ethereum',
        POLKADOT: 'Polkadot',
        KUSAMA: 'Kusama',
        PARACHAIN: 'Parachain',
        BELIZE_CHAIN: 'BelizeChain',
    }
    
    @classmethod
    def to_name(cls, index: int) -> str:
        """Convert index to destination name."""
        return cls._INDEX_TO_NAME.get(index, 'Unknown')
    
    @classmethod
    def validate(cls, index: int) -> bool:
        """Validate destination index is in range."""
        return 0 <= index <= 4


# Convenience validation functions
def validate_all_indices(
    backend_index: int,
    status_index: int,
    verification_index: int,
    achievement_index: int
) -> tuple[bool, str]:
    """
    Validate all quantum pallet indices at once.
    
    Args:
        backend_index: Backend index to validate
        status_index: Job status index to validate
        verification_index: Verification status index to validate
        achievement_index: Achievement type index to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not QuantumBackendIndex.validate(backend_index):
        return False, f"Invalid backend index {backend_index}. Must be 0-7."
    
    if not JobStatusIndex.validate(status_index):
        return False, f"Invalid status index {status_index}. Must be 0-4."
    
    if not VerificationStatusIndex.validate(verification_index):
        return False, f"Invalid verification index {verification_index}. Must be 0-3."
    
    if not AchievementTypeIndex.validate(achievement_index):
        return False, f"Invalid achievement index {achievement_index}. Must be 0-11."
    
    return True, ""


def get_index_summary() -> str:
    """
    Get human-readable summary of all index mappings.
    
    Returns:
        Formatted string with all index mappings
    """
    return f"""
Quantum Pallet Index Mappings
==============================

Backends (0-7):
{chr(10).join(f"  {i}: {QuantumBackendIndex.to_name(i)}" for i in range(8))}

Job Status (0-4):
{chr(10).join(f"  {i}: {JobStatusIndex.to_name(i)}" for i in range(5))}

Verification Status (0-3):
{chr(10).join(f"  {i}: {VerificationStatusIndex.to_name(i)}" for i in range(4))}

Achievement Types (0-11):
{chr(10).join(f"  {i}: {AchievementTypeIndex.to_name(i)}" for i in range(12))}

Verification Votes (0-2):
{chr(10).join(f"  {i}: {VerificationVoteIndex.to_name(i)}" for i in range(3))}

Chain Destinations (0-4):
{chr(10).join(f"  {i}: {ChainDestinationIndex.to_name(i)}" for i in range(5))}
"""


if __name__ == "__main__":
    # Print index summary when run directly
    print(get_index_summary())
    
    # Test validation
    print("\nValidation Tests:")
    print(f"Backend 4 (Qiskit): {QuantumBackendIndex.validate(4)} ✅")
    print(f"Backend 10 (invalid): {QuantumBackendIndex.validate(10)} ❌")
    print(f"Status 2 (Completed): {JobStatusIndex.validate(2)} ✅")
    print(f"Achievement 11 (Custom): {AchievementTypeIndex.validate(11)} ✅")
    
    # Test string conversion
    print("\nString Conversion Tests:")
    print(f"'azure_ionq' → {QuantumBackendIndex.from_string('azure_ionq')} (AzureIonQ)")
    print(f"'qiskit' → {QuantumBackendIndex.from_string('qiskit')} (Qiskit)")
    print(f"'unknown' → {QuantumBackendIndex.from_string('unknown')} (Other)")
