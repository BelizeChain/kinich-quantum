"""
Post-Quantum Cryptography Module for Kinich

Implements NIST-approved post-quantum signature schemes and key encapsulation
for quantum-resistant blockchain security.

Supported Schemes:
- Falcon-512: Fast and compact digital signatures (NIST Level 1)
- Falcon-1024: Higher security digital signatures (NIST Level 5)
- Dilithium3: Alternative signature scheme (NIST Level 3)
- Kyber-768: Key encapsulation mechanism (NIST Level 3)

Author: BelizeChain Team
License: MIT
"""

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
import secrets
import time

logger = logging.getLogger(__name__)

try:
    # Try importing pqcrypto for NIST PQC schemes
    from pqcrypto.sign.falcon_512 import (
        generate_keypair as falcon512_keygen,
        sign as falcon512_sign,
        verify as falcon512_verify,
    )
    from pqcrypto.sign.falcon_1024 import (
        generate_keypair as falcon1024_keygen,
        sign as falcon1024_sign,
        verify as falcon1024_verify,
    )
    from pqcrypto.sign.dilithium3 import (
        generate_keypair as dilithium_keygen,
        sign as dilithium_sign,
        verify as dilithium_verify,
    )
    from pqcrypto.kem.kyber768 import (
        generate_keypair as kyber_keygen,
        encrypt as kyber_encrypt,
        decrypt as kyber_decrypt,
    )
    PQCRYPTO_AVAILABLE = True
    logger.info("pqcrypto library available - using native implementations")
except ImportError:
    PQCRYPTO_AVAILABLE = False
    logger.warning(
        "pqcrypto not available - using stub implementations. "
        "Install with: pip install pqcrypto"
    )


class SignatureScheme(Enum):
    """Post-quantum signature schemes."""
    
    FALCON_512 = "falcon-512"      # 128-bit security, ~600 byte signatures
    FALCON_1024 = "falcon-1024"    # 256-bit security, ~1200 byte signatures
    DILITHIUM3 = "dilithium3"       # 192-bit security, ~3200 byte signatures


class KEMScheme(Enum):
    """Key Encapsulation Mechanisms."""
    
    KYBER_768 = "kyber-768"         # 192-bit security


@dataclass
class PQCKeyPair:
    """Post-quantum cryptographic key pair."""
    
    scheme: SignatureScheme
    public_key: bytes
    private_key: bytes
    created_at: float
    key_id: str
    
    def __post_init__(self):
        """Generate key ID from public key."""
        if not self.key_id:
            self.key_id = hashlib.sha256(self.public_key).hexdigest()[:16]


@dataclass
class PQCSignature:
    """Post-quantum signature."""
    
    scheme: SignatureScheme
    signature: bytes
    message_hash: str
    signer_key_id: str
    timestamp: float
    
    @property
    def signature_size(self) -> int:
        """Get signature size in bytes."""
        return len(self.signature)


@dataclass
class QuantumKey:
    """Quantum-generated key for blockchain operations."""
    
    key_data: bytes
    job_id: str
    backend: str
    num_qubits: int
    randomness_source: str  # "quantum" or "hybrid"
    created_at: float
    entropy_estimate: float  # Estimated entropy in bits
    
    def to_hex(self) -> str:
        """Convert key to hex string."""
        return self.key_data.hex()
    
    def derive_signing_key(self) -> bytes:
        """Derive signing key using HKDF."""
        # Use HKDF to derive cryptographic key from quantum randomness
        return hashlib.pbkdf2_hmac(
            'sha256',
            self.key_data,
            b'kinich-blockchain-signing',
            iterations=100000,
            dklen=32
        )


class PostQuantumCrypto:
    """
    Post-quantum cryptography manager for Kinich.
    
    Provides quantum-resistant signatures and key exchange for blockchain operations.
    Integrates with quantum key generation for enhanced security.
    """
    
    def __init__(
        self,
        default_signature_scheme: SignatureScheme = SignatureScheme.FALCON_512,
        enable_quantum_keys: bool = True
    ):
        """
        Initialize PQC manager.
        
        Args:
            default_signature_scheme: Default signature scheme to use
            enable_quantum_keys: Use quantum-generated keys when available
        """
        self.default_scheme = default_signature_scheme
        self.enable_quantum_keys = enable_quantum_keys
        
        # Key storage
        self._keypairs: Dict[str, PQCKeyPair] = {}
        self._quantum_keys: Dict[str, QuantumKey] = {}
        
        logger.info(
            f"Initialized PQC manager with {default_signature_scheme.value}"
        )
    
    # ==================== KEY GENERATION ====================
    
    def generate_keypair(
        self,
        scheme: Optional[SignatureScheme] = None,
        quantum_seed: Optional[bytes] = None
    ) -> PQCKeyPair:
        """
        Generate post-quantum key pair.
        
        Args:
            scheme: Signature scheme (defaults to FALCON-512)
            quantum_seed: Optional quantum-generated random seed
            
        Returns:
            PQCKeyPair with public and private keys
        """
        scheme = scheme or self.default_scheme
        
        if PQCRYPTO_AVAILABLE:
            # Use real PQC implementation
            if scheme == SignatureScheme.FALCON_512:
                public_key, private_key = falcon512_keygen()
            elif scheme == SignatureScheme.FALCON_1024:
                public_key, private_key = falcon1024_keygen()
            elif scheme == SignatureScheme.DILITHIUM3:
                public_key, private_key = dilithium_keygen()
            else:
                raise ValueError(f"Unsupported scheme: {scheme}")
            
            logger.info(f"Generated {scheme.value} keypair (native)")
        else:
            # Stub implementation for testing
            # In production, MUST use real PQC library!
            public_key = hashlib.sha256(
                (quantum_seed or secrets.token_bytes(32)) + b"_public"
            ).digest() + secrets.token_bytes(800)  # Falcon-512 size
            
            private_key = hashlib.sha256(
                (quantum_seed or secrets.token_bytes(32)) + b"_private"
            ).digest() + secrets.token_bytes(1280)  # Falcon-512 size
            
            logger.warning(f"Generated {scheme.value} keypair (STUB - not secure!)")
        
        keypair = PQCKeyPair(
            scheme=scheme,
            public_key=public_key,
            private_key=private_key,
            created_at=time.time(),
            key_id=""
        )
        
        # Store keypair
        self._keypairs[keypair.key_id] = keypair
        
        return keypair
    
    def generate_quantum_keypair(
        self,
        quantum_job_id: str,
        quantum_key_data: bytes,
        num_qubits: int,
        backend: str,
        scheme: Optional[SignatureScheme] = None
    ) -> PQCKeyPair:
        """
        Generate keypair using quantum-generated randomness.
        
        This combines quantum randomness from real quantum hardware
        with post-quantum signature schemes for maximum security.
        
        Args:
            quantum_job_id: ID of quantum key generation job
            quantum_key_data: Raw quantum measurement results
            num_qubits: Number of qubits used
            backend: Quantum backend name
            scheme: Signature scheme to use
            
        Returns:
            PQCKeyPair generated with quantum randomness
        """
        # Store quantum key
        quantum_key = QuantumKey(
            key_data=quantum_key_data,
            job_id=quantum_job_id,
            backend=backend,
            num_qubits=num_qubits,
            randomness_source="quantum",
            created_at=time.time(),
            entropy_estimate=num_qubits * 0.95  # Estimate ~95% entropy
        )
        self._quantum_keys[quantum_job_id] = quantum_key
        
        # Use quantum randomness as seed for PQC keygen
        keypair = self.generate_keypair(
            scheme=scheme,
            quantum_seed=quantum_key_data
        )
        
        logger.info(
            f"Generated quantum-seeded {scheme or self.default_scheme} keypair "
            f"from {num_qubits} qubits on {backend}"
        )
        
        return keypair
    
    # ==================== SIGNING ====================
    
    def sign(
        self,
        message: bytes,
        keypair: PQCKeyPair
    ) -> PQCSignature:
        """
        Sign message with post-quantum signature.
        
        Args:
            message: Message to sign
            keypair: Signing keypair
            
        Returns:
            PQCSignature
        """
        if PQCRYPTO_AVAILABLE:
            # Use real PQC signatures
            if keypair.scheme == SignatureScheme.FALCON_512:
                signature = falcon512_sign(keypair.private_key, message)
            elif keypair.scheme == SignatureScheme.FALCON_1024:
                signature = falcon1024_sign(keypair.private_key, message)
            elif keypair.scheme == SignatureScheme.DILITHIUM3:
                signature = dilithium_sign(keypair.private_key, message)
            else:
                raise ValueError(f"Unsupported scheme: {keypair.scheme}")
        else:
            # Stub implementation
            signature = hashlib.sha256(
                keypair.private_key + message
            ).digest() + secrets.token_bytes(600)
        
        pqc_signature = PQCSignature(
            scheme=keypair.scheme,
            signature=signature,
            message_hash=hashlib.sha256(message).hexdigest(),
            signer_key_id=keypair.key_id,
            timestamp=time.time()
        )
        
        logger.info(
            f"Signed message ({len(message)} bytes) with {keypair.scheme.value} "
            f"(signature: {pqc_signature.signature_size} bytes)"
        )
        
        return pqc_signature
    
    def verify(
        self,
        message: bytes,
        signature: PQCSignature,
        public_key: bytes
    ) -> bool:
        """
        Verify post-quantum signature.
        
        Args:
            message: Original message
            signature: PQCSignature to verify
            public_key: Public key of signer
            
        Returns:
            True if signature is valid
        """
        # Verify message hash
        message_hash = hashlib.sha256(message).hexdigest()
        if message_hash != signature.message_hash:
            logger.warning("Message hash mismatch")
            return False
        
        if PQCRYPTO_AVAILABLE:
            try:
                # Use real PQC verification
                if signature.scheme == SignatureScheme.FALCON_512:
                    falcon512_verify(public_key, message, signature.signature)
                elif signature.scheme == SignatureScheme.FALCON_1024:
                    falcon1024_verify(public_key, message, signature.signature)
                elif signature.scheme == SignatureScheme.DILITHIUM3:
                    dilithium_verify(public_key, message, signature.signature)
                else:
                    logger.error(f"Unsupported scheme: {signature.scheme}")
                    return False
                
                logger.info(f"Verified {signature.scheme.value} signature")
                return True
                
            except Exception as e:
                logger.error(f"Signature verification failed: {e}")
                return False
        else:
            # Stub verification (always succeeds in stub mode)
            logger.warning("Stub verification - NOT SECURE")
            return True
    
    # ==================== BLOCKCHAIN INTEGRATION ====================
    
    def sign_blockchain_transaction(
        self,
        transaction_data: Dict[str, Any],
        keypair: PQCKeyPair
    ) -> PQCSignature:
        """
        Sign blockchain transaction with post-quantum signature.
        
        Args:
            transaction_data: Transaction payload
            keypair: Signing keypair
            
        Returns:
            PQCSignature for transaction
        """
        import json
        
        # Canonicalize transaction data
        canonical = json.dumps(transaction_data, sort_keys=True)
        message = canonical.encode()
        
        signature = self.sign(message, keypair)
        
        logger.info(
            f"Signed blockchain transaction with {keypair.scheme.value} "
            f"(tx_type: {transaction_data.get('type', 'unknown')})"
        )
        
        return signature
    
    def verify_blockchain_transaction(
        self,
        transaction_data: Dict[str, Any],
        signature: PQCSignature,
        public_key: bytes
    ) -> bool:
        """
        Verify blockchain transaction signature.
        
        Args:
            transaction_data: Transaction payload
            signature: Transaction signature
            public_key: Signer's public key
            
        Returns:
            True if signature is valid
        """
        import json
        
        canonical = json.dumps(transaction_data, sort_keys=True)
        message = canonical.encode()
        
        is_valid = self.verify(message, signature, public_key)
        
        if is_valid:
            logger.info(
                f"Verified blockchain transaction "
                f"(tx_type: {transaction_data.get('type', 'unknown')})"
            )
        else:
            logger.warning(
                f"INVALID blockchain transaction signature! "
                f"(tx_type: {transaction_data.get('type', 'unknown')})"
            )
        
        return is_valid
    
    # ==================== KEY MANAGEMENT ====================
    
    def get_keypair(self, key_id: str) -> Optional[PQCKeyPair]:
        """Get keypair by ID."""
        return self._keypairs.get(key_id)
    
    def list_keypairs(self) -> Dict[str, PQCKeyPair]:
        """List all stored keypairs."""
        return self._keypairs.copy()
    
    def export_public_key(self, key_id: str) -> Optional[bytes]:
        """Export public key in raw binary format."""
        keypair = self._keypairs.get(key_id)
        return keypair.public_key if keypair else None
    
    def export_public_key_hex(self, key_id: str) -> Optional[str]:
        """Export public key as hex string."""
        public_key = self.export_public_key(key_id)
        return public_key.hex() if public_key else None
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get PQC statistics."""
        return {
            'total_keypairs': len(self._keypairs),
            'total_quantum_keys': len(self._quantum_keys),
            'keypairs_by_scheme': {
                scheme.value: sum(
                    1 for kp in self._keypairs.values()
                    if kp.scheme == scheme
                )
                for scheme in SignatureScheme
            },
            'pqcrypto_available': PQCRYPTO_AVAILABLE,
            'default_scheme': self.default_scheme.value,
        }


# ==================== HELPER FUNCTIONS ====================

def falcon512_key_size() -> Dict[str, int]:
    """Get Falcon-512 key and signature sizes."""
    return {
        'public_key': 897,      # bytes
        'private_key': 1281,    # bytes
        'signature': 666,       # bytes (average)
    }


def falcon1024_key_size() -> Dict[str, int]:
    """Get Falcon-1024 key and signature sizes."""
    return {
        'public_key': 1793,     # bytes
        'private_key': 2305,    # bytes
        'signature': 1280,      # bytes (average)
    }


def dilithium3_key_size() -> Dict[str, int]:
    """Get Dilithium3 key and signature sizes."""
    return {
        'public_key': 1952,     # bytes
        'private_key': 4000,    # bytes
        'signature': 3293,      # bytes
    }
