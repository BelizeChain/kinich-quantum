"""
Quantum Node for Kinich

Core quantum computing node that manages job execution,
resource allocation, and communication with the blockchain.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import uuid
from enum import Enum

from .jobs import (
    QuantumJob,
    JobStatus,
    JobPriority,
    QuantumJobResult,
)
from ..queue.job_scheduler import QuantumJobQueue

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of quantum node."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class NodeConfig:
    """Configuration for quantum node."""
    
    # Node identity
    node_id: str = field(default_factory=lambda: f"kinich-{uuid.uuid4().hex[:8]}")
    node_name: str = field(default="Kinich Quantum Node")
    operator_address: Optional[str] = field(default=None)
    
    # Blockchain connection
    rpc_url: str = field(default="ws://localhost:9944")
    submit_results_to_chain: bool = field(default=True)
    proof_of_quantum_work: bool = field(default=True)
    
    # Resource limits
    max_concurrent_jobs: int = field(default=10)
    max_queue_size: int = field(default=100)
    max_job_runtime: int = field(default=600)  # seconds
    
    # Networking
    enable_p2p: bool = field(default=True)
    p2p_port: int = field(default=9950)
    
    # Monitoring
    enable_metrics: bool = field(default=True)
    metrics_port: int = field(default=9951)
    log_level: str = field(default="INFO")


class QuantumNode:
    """
    Kinich Quantum Node - distributed quantum computing node.
    
    Responsibilities:
    - Accept quantum jobs from blockchain
    - Execute jobs on available quantum backends
    - Submit results back to blockchain
    - Participate in Proof of Quantum Work consensus
    - Share resources with other nodes (P2P)
    - Track quantum contributions to Community pallet SRS
    """
    
    def __init__(
        self,
        config: Optional[NodeConfig] = None,
        adapter_registry=None,
        job_queue: Optional[QuantumJobQueue] = None,
        enable_community_tracking: bool = True,
    ):
        """
        Initialize quantum node.
        
        Args:
            config: Node configuration
            adapter_registry: Adapter registry for quantum backends
            job_queue: Redis-based job queue (if None, uses local queue)
            enable_community_tracking: Enable SRS tracking (default: True)
        """
        self.config = config or NodeConfig()
        self.adapter_registry = adapter_registry
        self.job_queue = job_queue  # Redis queue for production
        
        # Node state
        self.status = NodeStatus.INITIALIZING
        self.node_id = self.config.node_id
        
        # Job management (local fallback if Redis not available)
        self._job_queue: List[QuantumJob] = []
        self._active_jobs: Dict[str, QuantumJob] = {}
        self._completed_jobs: Dict[str, QuantumJobResult] = {}
        
        # Statistics
        self._total_jobs_received = 0
        self._total_jobs_completed = 0
        self._total_jobs_failed = 0
        self._total_execution_time = 0.0
        
        # Blockchain connection
        self._blockchain_client = None
        
        # Community pallet integration for SRS tracking
        self.enable_community_tracking = enable_community_tracking
        self._community_tracker = None
        if enable_community_tracking:
            try:
                from kinich.blockchain.community_tracker import QuantumCommunityTracker
                self._community_tracker = QuantumCommunityTracker(
                    websocket_url="ws://127.0.0.1:9944",
                    enabled=True
                )
                logger.info("Community SRS tracking initialized for quantum node")
            except Exception as e:
                logger.warning(f"Failed to initialize QuantumCommunityTracker: {e}")
                self.enable_community_tracking = False
        
        # Event handlers
        self._job_received_handlers: List[Callable] = []
        self._job_completed_handlers: List[Callable] = []
        
        logger.info(
            f"Initialized quantum node: {self.node_id} "
            f"({self.config.node_name})"
        )
    
    async def start(self) -> None:
        """Start quantum node."""
        logger.info(f"Starting quantum node {self.node_id}...")
        
        # Initialize blockchain connection
        if self.config.submit_results_to_chain:
            await self._connect_to_blockchain()
        
        # Initialize community SRS tracker
        if self.enable_community_tracking and self._community_tracker:
            await self._community_tracker.connect()
            logger.info("Community SRS tracker connected")
        
        # Initialize P2P network
        if self.config.enable_p2p:
            await self._start_p2p_network()
        
        # Start job processor
        asyncio.create_task(self._process_job_queue())
        
        # Start metrics server
        if self.config.enable_metrics:
            await self._start_metrics_server()
        
        self.status = NodeStatus.READY
        logger.info(f"Quantum node {self.node_id} is ready")
    
    async def stop(self) -> None:
        """Stop quantum node."""
        logger.info(f"Stopping quantum node {self.node_id}...")
        
        self.status = NodeStatus.OFFLINE
        
        # Wait for active jobs to complete
        if self._active_jobs:
            logger.info(f"Waiting for {len(self._active_jobs)} active jobs...")
            await self._wait_for_active_jobs()
        
        # Disconnect from blockchain
        if self._blockchain_client:
            await self._disconnect_from_blockchain()
        
        logger.info(f"Quantum node {self.node_id} stopped")
    
    def submit_job(self, job: QuantumJob) -> bool:
        """
        Submit job to node queue.
        
        Args:
            job: Quantum job to execute
        
        Returns:
            True if accepted, False if rejected
        """
        # Check if we can handle this job
        if not self._can_handle_job(job):
            logger.warning(f"Cannot handle job {job.get_job_id()}")
            return False
        
        # Use Redis queue if available, otherwise local queue
        if self.job_queue:
            try:
                # Submit to Redis-based priority queue
                asyncio.create_task(self.job_queue.enqueue(job))
                self._total_jobs_received += 1
                logger.info(
                    f"Job {job.get_job_id()} submitted to Redis queue "
                    f"(priority: {job.priority.name})"
                )
            except Exception as e:
                logger.error(f"Failed to enqueue job to Redis: {e}")
                return False
        else:
            # Local queue fallback
            if len(self._job_queue) >= self.config.max_queue_size:
                logger.warning(f"Job queue full, rejecting job {job.get_job_id()}")
                return False
            
            self._job_queue.append(job)
            self._total_jobs_received += 1
            self._job_queue.sort(key=lambda j: j.priority.value)
            
            logger.info(
                f"Job {job.get_job_id()} queued locally "
                f"(priority: {job.priority.name}, queue size: {len(self._job_queue)})"
            )
        
        # Trigger job received handlers
        for handler in self._job_received_handlers:
            try:
                handler(job)
            except Exception as e:
                logger.error(f"Job received handler failed: {e}")
        
        return True
    
    def _can_handle_job(self, job: QuantumJob) -> bool:
        """Check if node can handle job."""
        # Check if we have suitable adapter
        if self.adapter_registry is None:
            return False
        
        # Check qubit requirements
        if hasattr(job, 'estimated_qubits'):
            suitable = self.adapter_registry.find_suitable_adapter(
                min_qubits=job.estimated_qubits
            )
            return suitable is not None
        
        return True
    
    async def _process_job_queue(self) -> None:
        """Process jobs from queue."""
        while self.status != NodeStatus.OFFLINE:
            try:
                # Check if we can take more jobs
                if len(self._active_jobs) >= self.config.max_concurrent_jobs:
                    await asyncio.sleep(1)
                    continue
                
                # Get next job from queue
                if not self._job_queue:
                    await asyncio.sleep(0.5)
                    continue
                
                job = self._job_queue.pop(0)
                
                # Execute job
                asyncio.create_task(self._execute_job(job))
                
            except Exception as e:
                logger.error(f"Job queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_job(self, job: QuantumJob) -> None:
        """Execute a single job."""
        job_id = job.get_job_id()
        
        logger.info(f"Executing job {job_id} (type: {job.job_type.name})")
        
        # Mark as active
        self._active_jobs[job_id] = job
        job.mark_started()
        
        try:
            # Build quantum circuit
            circuit = await self._build_circuit_for_job(job)
            
            if circuit is None:
                raise RuntimeError("Failed to build circuit")
            
            # Execute on best available backend
            result = await self._execute_circuit(circuit, job)
            
            # Mark as completed
            job.mark_completed(result)
            self._completed_jobs[job_id] = result
            self._total_jobs_completed += 1
            
            # Update execution time
            if job.get_execution_time():
                self._total_execution_time += job.get_execution_time()
            
            # Submit to blockchain
            if self.config.submit_results_to_chain:
                await self._submit_result_to_chain(job, result)
            
            # Record quantum job completion in Community pallet for SRS tracking
            if self.enable_community_tracking and self._community_tracker and job.submitter_account:
                try:
                    await self._community_tracker.record_quantum_job_completion(
                        account_id=job.submitter_account,
                        job_id=job_id,
                        backend_name=result.backend_name if result else "unknown",
                        shots=job.shots,
                        circuit_depth=circuit.depth() if hasattr(circuit, 'depth') else 0,
                        success=result is not None and result.success,
                        execution_time_seconds=job.get_execution_time() or 0.0,
                        error_mitigation_used=job.enable_error_mitigation
                    )
                    logger.info(f"Quantum job SRS recorded for {job.submitter_account[:8]}")
                except Exception as e:
                    logger.warning(f"Failed to record quantum SRS (continuing): {e}")
            
            # Trigger completion handlers
            for handler in self._job_completed_handlers:
                try:
                    handler(job, result)
                except Exception as e:
                    logger.error(f"Job completed handler failed: {e}")
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            job.mark_failed(str(e))
            self._total_jobs_failed += 1
        
        finally:
            # Remove from active jobs
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
    
    async def _build_circuit_for_job(self, job: QuantumJob) -> Optional[Any]:
        """Build quantum circuit for job."""
        # This will be implemented based on job type
        # For now, return None to trigger error
        logger.warning(f"Circuit building not implemented for {job.job_type.name}")
        return None
    
    async def _execute_circuit(self, circuit: Any, job: QuantumJob) -> QuantumJobResult:
        """Execute circuit on quantum backend."""
        if self.adapter_registry is None:
            raise RuntimeError("No adapter registry available")
        
        # Execute on best adapter
        result_dict = self.adapter_registry.execute_on_best_adapter(
            circuit=circuit,
            shots=job.shots,
            backend_preference=job.backend_preference
        )
        
        # Convert to QuantumJobResult
        return QuantumJobResult(
            counts=result_dict.get('counts', {}),
            shots_used=result_dict.get('shots', job.shots),
            backend_used=result_dict.get('adapter_used', 'unknown'),
            execution_time=result_dict.get('execution_time', 0.0),
        )
    
    async def _connect_to_blockchain(self) -> None:
        """Connect to BelizeChain."""
        logger.info(f"Connecting to blockchain: {self.config.rpc_url}")
        
        try:
            # Placeholder for Substrate connection
            # from substrateinterface import SubstrateInterface
            # self._blockchain_client = SubstrateInterface(url=self.config.rpc_url)
            
            logger.info("Connected to blockchain")
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
    
    async def _disconnect_from_blockchain(self) -> None:
        """Disconnect from blockchain."""
        if self._blockchain_client:
            logger.info("Disconnecting from blockchain")
            # self._blockchain_client.close()
            self._blockchain_client = None
    
    async def _submit_result_to_chain(
        self,
        job: QuantumJob,
        result: QuantumJobResult
    ) -> None:
        """Submit job result to blockchain."""
        logger.info(f"Submitting result for job {job.get_job_id()} to chain")
        
        # Placeholder for blockchain submission
        # This will call the appropriate pallet extrinsic
    
    async def _start_p2p_network(self) -> None:
        """Start P2P networking."""
        logger.info(f"Starting P2P network on port {self.config.p2p_port}")
        # Placeholder for P2P implementation
    
    async def _start_metrics_server(self) -> None:
        """Start Prometheus metrics server."""
        logger.info(f"Starting metrics server on port {self.config.metrics_port}")
        # Placeholder for metrics implementation
    
    async def _wait_for_active_jobs(self, timeout: int = 300) -> None:
        """Wait for active jobs to complete."""
        start_time = datetime.now()
        
        while self._active_jobs:
            await asyncio.sleep(1)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for jobs, {len(self._active_jobs)} still active")
                break
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status."""
        return {
            'node_id': self.node_id,
            'node_name': self.config.node_name,
            'status': self.status.value,
            'operator_address': self.config.operator_address,
            'jobs': {
                'queued': len(self._job_queue),
                'active': len(self._active_jobs),
                'completed': len(self._completed_jobs),
            },
            'statistics': {
                'total_received': self._total_jobs_received,
                'total_completed': self._total_jobs_completed,
                'total_failed': self._total_jobs_failed,
                'success_rate': (
                    self._total_jobs_completed / self._total_jobs_received
                    if self._total_jobs_received > 0 else 0.0
                ),
                'total_execution_time': self._total_execution_time,
            },
            'configuration': {
                'max_concurrent_jobs': self.config.max_concurrent_jobs,
                'max_queue_size': self.config.max_queue_size,
                'blockchain_enabled': self.config.submit_results_to_chain,
                'p2p_enabled': self.config.enable_p2p,
            }
        }
    
    def on_job_received(self, handler: Callable) -> None:
        """Register job received handler."""
        self._job_received_handlers.append(handler)
    
    def on_job_completed(self, handler: Callable) -> None:
        """Register job completed handler."""
        self._job_completed_handlers.append(handler)
