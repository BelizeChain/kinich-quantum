"""
Cross-chain Bridge Type Definitions

Type definitions for cross-chain quantum result and NFT bridging.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class ChainType(Enum):
    """Supported blockchain types."""
    BELIZECHAIN = "belizechain"
    ETHEREUM = "ethereum"
    POLKADOT = "polkadot"
    KUSAMA = "kusama"
    PARACHAIN = "parachain"


class BridgeStatus(Enum):
    """Status of bridge operation."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERSED = "reversed"


class AssetType(Enum):
    """Type of asset being bridged."""
    QUANTUM_RESULT = "quantum_result"
    ACHIEVEMENT_NFT = "achievement_nft"
    QUANTUM_TOKEN = "quantum_token"


@dataclass
class BridgeDestination:
    """Destination chain for bridging."""
    chain_type: ChainType
    chain_id: Optional[int] = None  # For Ethereum, parachain ID, etc.
    account: str = ""  # Destination account address
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chain_type': self.chain_type.value,
            'chain_id': self.chain_id,
            'account': self.account
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BridgeDestination':
        """Create from dictionary."""
        return cls(
            chain_type=ChainType(data['chain_type']),
            chain_id=data.get('chain_id'),
            account=data.get('account', '')
        )


@dataclass
class BridgeTransaction:
    """Record of a cross-chain bridge transaction."""
    tx_id: str
    source_chain: ChainType
    destination: BridgeDestination
    asset_type: AssetType
    asset_id: str  # Job ID, NFT ID, etc.
    status: BridgeStatus
    timestamp: datetime
    source_tx_hash: Optional[str] = None
    destination_tx_hash: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tx_id': self.tx_id,
            'source_chain': self.source_chain.value,
            'destination': self.destination.to_dict(),
            'asset_type': self.asset_type.value,
            'asset_id': self.asset_id,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'source_tx_hash': self.source_tx_hash,
            'destination_tx_hash': self.destination_tx_hash,
            'error_message': self.error_message,
            'metadata': self.metadata or {}
        }


@dataclass
class QuantumResultBridgeData:
    """Data for bridging quantum result."""
    job_id: str
    cid: str  # Content ID of result
    circuit_hash: bytes
    backend: str
    num_qubits: int
    circuit_depth: int
    accuracy: float
    result_hash: bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'cid': self.cid,
            'circuit_hash': self.circuit_hash.hex(),
            'backend': self.backend,
            'num_qubits': self.num_qubits,
            'circuit_depth': self.circuit_depth,
            'accuracy': self.accuracy,
            'result_hash': self.result_hash.hex()
        }


@dataclass
class NFTBridgeData:
    """Data for bridging achievement NFT."""
    nft_id: str
    achievement_type: str
    metadata_uri: str
    owner: str
    earned_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nft_id': self.nft_id,
            'achievement_type': self.achievement_type,
            'metadata_uri': self.metadata_uri,
            'owner': self.owner,
            'earned_date': self.earned_date.isoformat()
        }


@dataclass
class BridgeProof:
    """Cryptographic proof for bridge transaction."""
    proof_type: str  # "merkle", "signature", "zk"
    proof_data: bytes
    validator_signatures: List[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'proof_type': self.proof_type,
            'proof_data': self.proof_data.hex(),
            'validator_signatures': [
                sig.hex() for sig in (self.validator_signatures or [])
            ]
        }
