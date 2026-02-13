"""
Kinich Blockchain Integration Module

Provides blockchain connectivity for quantum computing operations on BelizeChain,
plus cross-chain bridging to Ethereum, Polkadot, and other networks.
"""

from kinich.blockchain.belizechain_adapter import (
    BelizeChainAdapter,
    JobStatus,
    VerificationStatus,
    QuantumBackend,
    AchievementType,
    QuantumJob,
    QuantumResult,
    QuantumAchievement,
)

from kinich.blockchain.contribution_tracker import (
    QuantumContributionTracker,
    ContributionRecord,
)

from kinich.blockchain.cross_chain_bridge import CrossChainBridge

from kinich.blockchain.bridge_types import (
    BridgeDestination,
    BridgeTransaction,
    BridgeStatus,
    ChainType,
    AssetType,
    QuantumResultBridgeData,
    NFTBridgeData,
    BridgeProof,
)

from kinich.blockchain.ethereum_bridge import EthereumBridge

from kinich.blockchain.polkadot_xcm import (
    PolkadotXCMBridge,
    XCMVersion,
    ParachainId,
    XCMMultiLocation,
    XCMMultiAsset,
)

__all__ = [
    # BelizeChain integration
    'BelizeChainAdapter',
    'JobStatus',
    'VerificationStatus',
    'QuantumBackend',
    'AchievementType',
    'QuantumJob',
    'QuantumResult',
    'QuantumAchievement',
    'QuantumContributionTracker',
    'ContributionRecord',
    
    # Cross-chain bridge
    'CrossChainBridge',
    'BridgeDestination',
    'BridgeTransaction',
    'BridgeStatus',
    'ChainType',
    'AssetType',
    'QuantumResultBridgeData',
    'NFTBridgeData',
    'BridgeProof',
    'EthereumBridge',
    
    # Polkadot XCM bridge
    'PolkadotXCMBridge',
    'XCMVersion',
    'ParachainId',
    'XCMMultiLocation',
    'XCMMultiAsset',
]
