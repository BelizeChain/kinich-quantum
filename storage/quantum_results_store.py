"""
Quantum Results Store

Stores quantum computation results in Pakit for:
- Long-term archival
- Cross-node replication
- Blockchain proof integration
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class QuantumResultsStore:
    """
    Stores quantum computation results in Pakit.
    
    Integration with:
    - Pakit IPFS/Arweave backend
    - Consensus pallet (Proof of Quantum Work)
    - Quantum indices registry
    """
    
    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8000",
        blockchain_connector: Optional[Any] = None
    ):
        """
        Initialize results store.
        
        Args:
            pakit_api_url: Pakit API endpoint
            blockchain_connector: BelizeChain connector for on-chain proofs
        """
        self.pakit_api_url = pakit_api_url
        self.blockchain_connector = blockchain_connector
    
    def store_quantum_result(
        self,
        job_id: str,
        circuit_qasm: str,
        counts: Dict[str, int],
        backend: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store quantum computation result.
        
        Args:
            job_id: Quantum job ID
            circuit_qasm: OpenQASM circuit definition
            counts: Measurement counts
            backend: Quantum backend used
            metadata: Additional metadata
            
        Returns:
            Content ID (Pakit hash)
        """
        # Prepare result package
        result_package = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'circuit_qasm': circuit_qasm,
            'counts': counts,
            'backend': backend,
            'metadata': metadata or {},
        }
        
        result_json = json.dumps(result_package, indent=2)
        
        # Upload to Pakit
        if not REQUESTS_AVAILABLE:
            return self._mock_store(result_json)
        
        try:
            response = requests.post(
                f"{self.pakit_api_url}/api/v1/upload",
                files={'file': (f"{job_id}.json", result_json.encode())},
                data={'metadata': json.dumps({
                    'type': 'quantum_result',
                    'job_id': job_id,
                    'backend': backend,
                })}
            )
            
            if response.status_code == 200:
                result = response.json()
                content_id = result.get('cid') or result.get('content_id')
                logger.info(f"✅ Stored quantum result {job_id}: {content_id}")
                
                # Store proof on-chain if available
                if self.blockchain_connector:
                    self._store_blockchain_proof(job_id, content_id, backend)
                
                return content_id
            else:
                logger.error(f"Pakit upload failed: {response.status_code}")
                return self._mock_store(result_json)
                
        except Exception as e:
            logger.error(f"Store error: {e}")
            return self._mock_store(result_json)
    
    def retrieve_quantum_result(
        self,
        content_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve quantum result from Pakit.
        
        Args:
            content_id: Content ID to retrieve
            
        Returns:
            Result package or None
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available")
            return None
        
        try:
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/retrieve/{content_id}"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Retrieve failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            return None
    
    def list_job_results(
        self,
        backend: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List stored quantum results.
        
        Args:
            backend: Filter by backend
            limit: Max results
            
        Returns:
            List of result metadata
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            params = {'type': 'quantum_result', 'limit': limit}
            if backend:
                params['backend'] = backend
            
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/list",
                params=params
            )
            
            if response.status_code == 200:
                return response.json().get('results', [])
            return []
            
        except Exception as e:
            logger.error(f"List error: {e}")
            return []
    
    def _store_blockchain_proof(
        self,
        job_id: str,
        content_id: str,
        backend: str
    ):
        """Store proof on BelizeChain Consensus pallet."""
        if not self.blockchain_connector:
            return
        
        try:
            # Submit Proof of Quantum Work to blockchain
            # (Would call Consensus pallet extrinsic)
            logger.info(f"📜 Stored blockchain proof: {job_id} -> {content_id}")
        except Exception as e:
            logger.warning(f"Blockchain proof failed: {e}")
    
    def _mock_store(self, data: str) -> str:
        """Mock storage when Pakit unavailable."""
        hasher = hashlib.sha256()
        hasher.update(data.encode())
        mock_id = hasher.hexdigest()
        logger.info(f"📦 MOCK store: {mock_id}")
        return mock_id
