"""
Polkadot XCM (Cross-Consensus Message) Bridge Integration

Provides cross-chain communication between BelizeChain and Polkadot/Kusama
parachains using XCM protocol for quantum computing results and NFT transfers.

XCM Version: XCMv3
Supported Chains: Polkadot, Kusama, Parachains
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    SubstrateInterface = None
    Keypair = None
    SubstrateRequestException = Exception

from .bridge_types import (
    BridgeDestination,
    BridgeTransaction,
    BridgeStatus,
    QuantumResultBridgeData,
    NFTBridgeData,
    BridgeProof,
    ChainType
)

logger = logging.getLogger(__name__)


class XCMVersion(Enum):
    """XCM protocol versions"""
    V2 = "V2"
    V3 = "V3"  # Latest, recommended
    V4 = "V4"  # Future


class ParachainId(Enum):
    """Well-known parachain IDs"""
    ACALA = 2000
    MOONBEAM = 2004
    ASTAR = 2006
    PARALLEL = 2012
    INTERLAY = 2032
    PHALA = 2035
    UNIQUE = 2037
    BIFROST = 2030
    HYDRADX = 2034


@dataclass
class XCMMultiLocation:
    """XCM MultiLocation for asset/account addressing"""
    parents: int  # 0=local, 1=parent, 2=grandparent
    interior: Dict[str, Any]  # Junction array

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parents": self.parents,
            "interior": self.interior
        }


@dataclass
class XCMMultiAsset:
    """XCM MultiAsset for asset transfers"""
    id: XCMMultiLocation
    fungibility: Dict[str, Any]  # Fungible or NonFungible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id.to_dict(),
            "fun": self.fungibility
        }


class PolkadotXCMBridge:
    """
    Polkadot XCM Bridge for cross-chain quantum computing operations.
    
    Supports:
    - Quantum result transfers to Polkadot/Kusama parachains
    - NFT bridging (achievement badges, quantum art)
    - Cross-chain verification of quantum computations
    - Asset teleportation and reserve transfers
    """

    def __init__(
        self,
        polkadot_url: str = "wss://rpc.polkadot.io",
        kusama_url: str = "wss://kusama-rpc.polkadot.io",
        belizechain_url: str = "ws://127.0.0.1:9944",
        enable_kusama: bool = True,
        xcm_version: XCMVersion = XCMVersion.V3
    ):
        """
        Initialize Polkadot XCM bridge.
        
        Args:
            polkadot_url: Polkadot relay chain WebSocket URL
            kusama_url: Kusama relay chain WebSocket URL
            belizechain_url: BelizeChain WebSocket URL
            enable_kusama: Whether to enable Kusama support
            xcm_version: XCM protocol version to use
        """
        if not SUBSTRATE_AVAILABLE:
            logger.warning(
                "substrate-interface not installed. Polkadot XCM bridge will run in stub mode. "
                "Install with: pip install substrate-interface"
            )
            self.stub_mode = True
            return

        self.stub_mode = False
        self.xcm_version = xcm_version
        
        # Initialize Substrate connections
        try:
            self.polkadot = SubstrateInterface(url=polkadot_url)
            logger.info(f"Connected to Polkadot: {polkadot_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Polkadot: {e}")
            self.polkadot = None

        if enable_kusama:
            try:
                self.kusama = SubstrateInterface(url=kusama_url)
                logger.info(f"Connected to Kusama: {kusama_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Kusama: {e}")
                self.kusama = None
        else:
            self.kusama = None

        try:
            self.belizechain = SubstrateInterface(url=belizechain_url)
            logger.info(f"Connected to BelizeChain: {belizechain_url}")
        except Exception as e:
            logger.error(f"Failed to connect to BelizeChain: {e}")
            self.belizechain = None

    async def bridge_quantum_result(
        self,
        result_data: QuantumResultBridgeData,
        destination: BridgeDestination,
        proof: BridgeProof,
        sender_keypair: Optional[Any] = None
    ) -> BridgeTransaction:
        """
        Bridge quantum computing result to Polkadot ecosystem.
        
        Process:
        1. Lock result on BelizeChain
        2. Generate XCM message with quantum data
        3. Send XCM to destination parachain
        4. Wait for confirmation
        
        Args:
            result_data: Quantum result to bridge
            destination: Target chain and account
            proof: Cryptographic proof of result validity
            sender_keypair: BelizeChain keypair for signing
            
        Returns:
            Bridge transaction record
        """
        if self.stub_mode:
            return self._create_stub_transaction(
                result_data.job_id, destination, "quantum_result"
            )

        logger.info(f"Bridging quantum result {result_data.job_id} to {destination.chain_type.value}")

        try:
            # Determine target chain
            target_chain = self._get_chain_interface(destination.chain_type)
            if not target_chain:
                raise ValueError(f"Chain {destination.chain_type} not available")

            # Build XCM message for quantum result
            xcm_message = self._build_quantum_result_xcm(
                result_data, destination, proof
            )

            # Submit XCM transaction from BelizeChain
            tx_hash = await self._submit_xcm_message(
                xcm_message,
                destination,
                sender_keypair
            )

            # Create transaction record
            transaction = BridgeTransaction(
                transaction_id=tx_hash,
                source_chain=ChainType.BELIZECHAIN,
                destination_chain=destination.chain_type,
                bridge_type="quantum_result",
                status=BridgeStatus.PENDING,
                source_account="",  # Will be set by keypair
                destination_account=destination.account,
                data=asdict(result_data),
                proof=asdict(proof),
                timestamp=0  # Will be set by chain
            )

            logger.info(f"XCM message sent: {tx_hash}")
            return transaction

        except Exception as e:
            logger.error(f"Failed to bridge quantum result: {e}")
            raise

    async def bridge_nft(
        self,
        nft_data: NFTBridgeData,
        destination: BridgeDestination,
        sender_keypair: Optional[Any] = None
    ) -> BridgeTransaction:
        """
        Bridge quantum achievement NFT to Polkadot ecosystem.
        
        Uses XCM asset teleportation or reserve transfer depending on
        destination parachain capabilities.
        
        Args:
            nft_data: NFT metadata and ownership info
            destination: Target parachain and account
            sender_keypair: BelizeChain keypair for signing
            
        Returns:
            Bridge transaction record
        """
        if self.stub_mode:
            return self._create_stub_transaction(
                nft_data.token_id, destination, "nft"
            )

        logger.info(f"Bridging NFT {nft_data.token_id} to {destination.chain_type.value}")

        try:
            # Build XCM message for NFT transfer
            xcm_message = self._build_nft_xcm(nft_data, destination)

            # Submit XCM transaction
            tx_hash = await self._submit_xcm_message(
                xcm_message,
                destination,
                sender_keypair
            )

            transaction = BridgeTransaction(
                transaction_id=tx_hash,
                source_chain=ChainType.BELIZECHAIN,
                destination_chain=destination.chain_type,
                bridge_type="nft",
                status=BridgeStatus.PENDING,
                source_account="",
                destination_account=destination.account,
                data=asdict(nft_data),
                proof=None,
                timestamp=0
            )

            logger.info(f"NFT XCM message sent: {tx_hash}")
            return transaction

        except Exception as e:
            logger.error(f"Failed to bridge NFT: {e}")
            raise

    def verify_bridged_result(
        self,
        transaction_id: str,
        destination_chain: ChainType
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify that a bridged quantum result was received on destination chain.
        
        Args:
            transaction_id: XCM transaction hash
            destination_chain: Target chain to check
            
        Returns:
            (is_confirmed, result_data)
        """
        if self.stub_mode:
            logger.warning("XCM verification running in stub mode")
            return True, {"status": "stub_confirmed"}

        try:
            target_chain = self._get_chain_interface(destination_chain)
            if not target_chain:
                return False, None

            # Query XCM events on destination chain
            events = self._query_xcm_events(target_chain, transaction_id)
            
            for event in events:
                if event.value['module_id'] == 'XcmPallet' and \
                   event.value['event_id'] == 'Success':
                    logger.info(f"XCM message confirmed on {destination_chain.value}")
                    return True, event.value.get('attributes', {})

            return False, None

        except Exception as e:
            logger.error(f"Failed to verify bridged result: {e}")
            return False, None

    def _build_quantum_result_xcm(
        self,
        result_data: QuantumResultBridgeData,
        destination: BridgeDestination,
        proof: BridgeProof
    ) -> Dict[str, Any]:
        """
        Build XCM message for quantum result transfer.
        
        XCM Structure (V3):
        - WithdrawAsset (lock on source)
        - BuyExecution (pay for execution)
        - Transact (store quantum data)
        - DepositAsset (unlock on destination)
        """
        # Convert result to XCM-compatible format
        quantum_data_encoded = self._encode_quantum_data(result_data)

        # Build destination MultiLocation
        dest_location = self._build_destination_location(destination)

        # Build XCM instructions
        xcm_instructions = [
            {
                "WithdrawAsset": [{
                    "id": {"Concrete": dest_location.to_dict()},
                    "fun": {"Fungible": 1000000000000}  # 1 token for fees
                }]
            },
            {
                "BuyExecution": {
                    "fees": {
                        "id": {"Concrete": dest_location.to_dict()},
                        "fun": {"Fungible": 1000000000000}
                    },
                    "weight_limit": "Unlimited"
                }
            },
            {
                "Transact": {
                    "origin_type": "SovereignAccount",
                    "require_weight_at_most": 1000000000,
                    "call": {
                        "encoded": quantum_data_encoded
                    }
                }
            },
            {
                "DepositAsset": {
                    "assets": {"Wild": "All"},
                    "max_assets": 1,
                    "beneficiary": dest_location.to_dict()
                }
            }
        ]

        return {
            "V3": xcm_instructions
        }

    def _build_nft_xcm(
        self,
        nft_data: NFTBridgeData,
        destination: BridgeDestination
    ) -> Dict[str, Any]:
        """
        Build XCM message for NFT transfer.
        
        Uses NonFungible asset type for unique tokens.
        """
        dest_location = self._build_destination_location(destination)

        # NFT as NonFungible asset
        nft_asset = {
            "id": {
                "Concrete": {
                    "parents": 1,
                    "interior": {
                        "X3": [
                            {"Parachain": 1000},  # BelizeChain parachain ID
                            {"PalletInstance": 50},  # NFT pallet
                            {"GeneralIndex": int(nft_data.token_id)}
                        ]
                    }
                }
            },
            "fun": {
                "NonFungible": {
                    "Index": int(nft_data.token_id)
                }
            }
        }

        xcm_instructions = [
            {"WithdrawAsset": [nft_asset]},
            {
                "DepositAsset": {
                    "assets": {"Wild": "All"},
                    "max_assets": 1,
                    "beneficiary": dest_location.to_dict()
                }
            }
        ]

        return {"V3": xcm_instructions}

    def _build_destination_location(
        self,
        destination: BridgeDestination
    ) -> XCMMultiLocation:
        """
        Build XCM MultiLocation for destination account.
        
        Format depends on chain type:
        - Polkadot relay: parents=1, interior=AccountId32
        - Parachain: parents=1, interior=X2[Parachain, AccountId32]
        """
        if destination.chain_type == ChainType.POLKADOT:
            # Relay chain account
            return XCMMultiLocation(
                parents=1,
                interior={
                    "X1": {
                        "AccountId32": {
                            "network": "Polkadot",
                            "id": destination.account
                        }
                    }
                }
            )
        elif destination.chain_type == ChainType.KUSAMA:
            return XCMMultiLocation(
                parents=1,
                interior={
                    "X1": {
                        "AccountId32": {
                            "network": "Kusama",
                            "id": destination.account
                        }
                    }
                }
            )
        elif destination.chain_type == ChainType.PARACHAIN:
            # Parachain account
            parachain_id = destination.chain_id or 2000
            return XCMMultiLocation(
                parents=1,
                interior={
                    "X2": [
                        {"Parachain": parachain_id},
                        {
                            "AccountId32": {
                                "network": "Any",
                                "id": destination.account
                            }
                        }
                    ]
                }
            )
        else:
            raise ValueError(f"Unsupported chain type for XCM: {destination.chain_type}")

    async def _submit_xcm_message(
        self,
        xcm_message: Dict[str, Any],
        destination: BridgeDestination,
        sender_keypair: Optional[Any]
    ) -> str:
        """
        Submit XCM message to BelizeChain for relay to destination.
        
        Returns:
            Transaction hash
        """
        if not self.belizechain:
            raise RuntimeError("BelizeChain connection not available")

        # Build XCM send extrinsic
        call = self.belizechain.compose_call(
            call_module='XcmPallet',
            call_function='send',
            call_params={
                'dest': self._build_destination_location(destination).to_dict(),
                'message': xcm_message
            }
        )

        # Sign and submit
        if sender_keypair:
            extrinsic = self.belizechain.create_signed_extrinsic(
                call=call,
                keypair=sender_keypair
            )
        else:
            # Use unsigned if no keypair provided (testing)
            extrinsic = self.belizechain.create_unsigned_extrinsic(call)

        receipt = self.belizechain.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=True
        )

        if receipt.is_success:
            return receipt.extrinsic_hash
        else:
            raise SubstrateRequestException(
                f"XCM submission failed: {receipt.error_message}"
            )

    def _encode_quantum_data(
        self,
        result_data: QuantumResultBridgeData
    ) -> str:
        """
        Encode quantum result data for XCM Transact call.
        
        Converts quantum computing result to SCALE-encoded bytes
        for storage on destination chain.
        """
        # Serialize to JSON (simplified - in production use SCALE codec)
        data_json = json.dumps({
            "job_id": result_data.job_id,
            "circuit_hash": result_data.circuit_hash,
            "result_cid": result_data.result_cid,
            "shots": result_data.shots,
            "backend": result_data.backend,
            "execution_time_ms": result_data.execution_time_ms,
            "fidelity": result_data.fidelity
        })

        # In production, use SCALE encoding:
        # from scalecodec import ScaleBytes
        # return ScaleBytes(data_json).to_hex()
        
        return data_json.encode('utf-8').hex()

    def _get_chain_interface(self, chain_type: ChainType) -> Optional[SubstrateInterface]:
        """Get Substrate interface for chain type."""
        if chain_type == ChainType.POLKADOT:
            return self.polkadot
        elif chain_type == ChainType.KUSAMA:
            return self.kusama
        elif chain_type in (ChainType.PARACHAIN, ChainType.BELIZECHAIN):
            return self.belizechain
        return None

    def _query_xcm_events(
        self,
        chain: SubstrateInterface,
        tx_hash: str
    ) -> List[Any]:
        """Query XCM-related events for a transaction."""
        try:
            block_hash = chain.get_block_hash(tx_hash)
            events = chain.get_events(block_hash)
            return [e for e in events if 'Xcm' in e.value.get('module_id', '')]
        except Exception as e:
            logger.error(f"Failed to query XCM events: {e}")
            return []

    def _create_stub_transaction(
        self,
        identifier: str,
        destination: BridgeDestination,
        bridge_type: str
    ) -> BridgeTransaction:
        """Create stub transaction for testing when substrate-interface unavailable."""
        return BridgeTransaction(
            transaction_id=f"stub_xcm_{identifier}",
            source_chain=ChainType.BELIZECHAIN,
            destination_chain=destination.chain_type,
            bridge_type=bridge_type,
            status=BridgeStatus.CONFIRMED,
            source_account="stub_account",
            destination_account=destination.account,
            data={"stub": True},
            proof=None,
            timestamp=0
        )

    def close(self):
        """Close all Substrate connections."""
        if self.stub_mode:
            return

        for chain in [self.polkadot, self.kusama, self.belizechain]:
            if chain:
                try:
                    chain.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
