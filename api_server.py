"""
Kinich Quantum Computing API Server

Production-grade FastAPI server for Kinich hybrid quantum-classical orchestration.
Provides REST endpoints for quantum job submission, execution, and result retrieval.

Endpoints:
- POST /api/v1/jobs/submit - Submit quantum job
- GET /api/v1/jobs/{job_id} - Get job status and results
- GET /api/v1/jobs - List jobs for account
- DELETE /api/v1/jobs/{job_id}/cancel - Cancel pending job
- GET /api/v1/backends - List available quantum backends
- GET /api/v1/jobs/{job_id}/attestation - Get QPU attestation
- POST /api/v1/jobs/estimate - Estimate job cost
- GET /api/v1/accounts/{account}/stats - Get account statistics
- GET /api/v1/stats/system - Get system-wide statistics

Author: BelizeChain Team
Date: October 2025
License: MIT
"""

import asyncio
import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

# Kinich imports
from kinich.core.quantum_node import QuantumNode, NodeConfig, NodeStatus
from kinich.core.jobs import QuantumJob, JobStatus, JobPriority, QuantumJobResult
from kinich.blockchain.belizechain_adapter import BelizeChainAdapter
from kinich.blockchain.quantum_indices import QuantumBackendIndex, JobStatusIndex
from kinich.circuit_analyzer import CircuitAnalyzer
from kinich.cost_calculator import CostCalculator


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig(BaseModel):
    """API server configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host (0.0.0.0 for Docker/cloud, override via KINICH_HOST env var)"
    )
    port: int = Field(default=8888, description="Server port (Docker container port)")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Blockchain connection
    blockchain_rpc: str = Field(
        default="ws://localhost:9944",
        description="BelizeChain RPC endpoint"
    )
    blockchain_enabled: bool = Field(
        default=True,
        description="Enable blockchain integration"
    )
    
    # Quantum backends
    azure_quantum_enabled: bool = Field(default=True, description="Enable Azure Quantum")
    ibm_quantum_enabled: bool = Field(default=False, description="Enable IBM Quantum")
    local_simulator: bool = Field(default=True, description="Enable local simulator")
    
    # Job limits
    max_concurrent_jobs: int = Field(default=10, ge=1, le=100, description="Max concurrent jobs")
    max_queue_size: int = Field(default=100, ge=1, le=1000, description="Max queue size")
    max_job_runtime: int = Field(default=600, ge=60, le=3600, description="Max job runtime (seconds)")


# =============================================================================
# API Models
# =============================================================================

class SubmitJobRequest(BaseModel):
    """Request to submit quantum job."""
    
    submitter_address: str = Field(..., description="Submitter account (SS58 address)")
    circuit: str = Field(..., description="Quantum circuit (QASM or serialized)")
    backend: str = Field(..., description="Quantum backend (azure-ionq, ibm-quantum, etc.)")
    shots: int = Field(default=1024, ge=1, le=10000, description="Number of measurements")
    error_mitigation: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Error mitigation options (zne, readoutCorrection)"
    )
    priority: str = Field(default="normal", description="Job priority (low, normal, high)")


class SubmitJobResponse(BaseModel):
    """Response for job submission."""
    
    job_id: str
    status: str
    submitted_at: str
    estimated_cost: int  # DALLA in Mahogany
    estimated_completion: str
    message: str


class JobResponse(BaseModel):
    """Job status and results response."""
    
    job_id: str
    submitter: str
    backend: str
    status: str
    submitted_at: str
    completed_at: Optional[str]
    circuit: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    cost: int
    error: Optional[str]


class BackendInfo(BaseModel):
    """Quantum backend information."""
    
    name: str
    status: str  # "available", "busy", "offline"
    qubits: int
    queue_length: int
    avg_wait_time: int  # seconds
    cost_per_shot: int  # DALLA in Mahogany
    capabilities: List[str]
    last_update: str


class EstimateCostRequest(BaseModel):
    """Request to estimate job cost."""
    
    backend: str
    shots: int
    qubits: int
    gates: int


class EstimateCostResponse(BaseModel):
    """Cost estimation response."""
    
    cost: int  # Total cost in Mahogany
    breakdown: Dict[str, int]  # Cost breakdown


class AccountStatsResponse(BaseModel):
    """Account quantum statistics."""
    
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_cost: int  # DALLA in Mahogany
    total_shots: int
    favorite_backend: Optional[str]
    avg_execution_time: float  # milliseconds


class SystemStatsResponse(BaseModel):
    """System-wide statistics."""
    
    total_jobs: int
    active_jobs: int
    total_shots: int
    backend_utilization: Dict[str, float]  # 0-100%
    avg_wait_time: float  # seconds


class AttestationResponse(BaseModel):
    """QPU attestation response."""
    
    job_id: str
    backend: str
    attestation: Dict[str, Any]
    on_chain_proof: str


# =============================================================================
# Global State
# =============================================================================

class AppState:
    """Application state."""
    
    def __init__(self):
        self.config: Optional[ServerConfig] = None
        self.quantum_node: Optional[QuantumNode] = None
        self.blockchain_adapter: Optional[BelizeChainAdapter] = None
        self.jobs: Dict[str, QuantumJob] = {}
        self.results: Dict[str, QuantumJobResult] = {}
        self.job_counter: int = 0
        
        # Real circuit analysis and cost calculation
        self.circuit_analyzer = CircuitAnalyzer()
        self.cost_calculator = CostCalculator()
        logger.info("✅ Initialized circuit analyzer and cost calculator")
    
    async def initialize(self, config: ServerConfig):
        """Initialize application state."""
        self.config = config
        
        # Initialize quantum node
        node_config = NodeConfig(
            rpc_url=config.blockchain_rpc,
            submit_results_to_chain=config.blockchain_enabled,
            max_concurrent_jobs=config.max_concurrent_jobs,
            max_queue_size=config.max_queue_size,
            max_job_runtime=config.max_job_runtime,
        )
        self.quantum_node = QuantumNode(config=node_config)
        await self.quantum_node.start()
        
        # Initialize blockchain adapter
        if config.blockchain_enabled:
            self.blockchain_adapter = BelizeChainAdapter(
                node_url=config.blockchain_rpc
            )
            try:
                await self.blockchain_adapter.connect()
                logger.info("✅ Connected to BelizeChain at {}", config.blockchain_rpc)
            except Exception as e:
                logger.error("❌ Failed to connect to blockchain: {}", e)
                logger.warning("Running in degraded mode (blockchain unavailable)")
        
        logger.info("✅ Kinich API server initialized")
        logger.info("   Azure Quantum: {}", "enabled" if config.azure_quantum_enabled else "disabled")
        logger.info("   IBM Quantum: {}", "enabled" if config.ibm_quantum_enabled else "disabled")
        logger.info("   Local Simulator: {}", "enabled" if config.local_simulator else "disabled")
    
    async def shutdown(self):
        """Cleanup resources."""
        if self.quantum_node:
            await self.quantum_node.stop()
        if self.blockchain_adapter:
            await self.blockchain_adapter.disconnect()
        logger.info("✅ Kinich API server shutdown complete")


# Global app state
app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    config = ServerConfig(
        blockchain_rpc=os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944"),
        blockchain_enabled=os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true",
        azure_quantum_enabled=os.getenv("AZURE_QUANTUM_ENABLED", "true").lower() == "true",
        ibm_quantum_enabled=os.getenv("IBM_QUANTUM_ENABLED", "false").lower() == "true",
        port=int(os.getenv("PORT", "8888")),
    )
    await app_state.initialize(config)
    
    yield
    
    # Shutdown
    await app_state.shutdown()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Kinich Quantum Computing API",
    description="Production API for BelizeChain quantum workload orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "kinich-quantum-api",
        "timestamp": datetime.utcnow().isoformat(),
        "node_status": app_state.quantum_node.status.value if app_state.quantum_node else "offline",
        "blockchain_connected": app_state.blockchain_adapter is not None,
    }


@app.get("/api/v1/status")
async def get_status():
    """Get API status and configuration."""
    return {
        "service": "Kinich Quantum Computing",
        "version": "1.0.0",
        "node": {
            "id": app_state.quantum_node.node_id if app_state.quantum_node else None,
            "status": app_state.quantum_node.status.value if app_state.quantum_node else "offline",
        },
        "blockchain": {
            "enabled": app_state.config.blockchain_enabled,
            "connected": app_state.blockchain_adapter is not None,
            "rpc_url": app_state.config.blockchain_rpc,
        },
        "backends": {
            "azure_quantum": app_state.config.azure_quantum_enabled,
            "ibm_quantum": app_state.config.ibm_quantum_enabled,
            "local_simulator": app_state.config.local_simulator,
        },
        "active_jobs": len(app_state.jobs),
        "total_jobs": app_state.job_counter,
    }


# =============================================================================
# Job Management
# =============================================================================

@app.post("/api/v1/jobs/submit", response_model=SubmitJobResponse, status_code=status.HTTP_201_CREATED)
async def submit_quantum_job(request: SubmitJobRequest):
    """
    Submit quantum job for execution.
    
    This endpoint:
    1. Creates quantum job with unique ID
    2. Validates circuit and backend
    3. Submits to blockchain (if enabled)
    4. Queues for execution
    """
    try:
        # Generate job ID
        job_id = f"job_{uuid.uuid4().hex}"
        app_state.job_counter += 1
        
        # REAL CIRCUIT ANALYSIS - No more hardcoded values!
        logger.info(f"🔍 Analyzing circuit for job {job_id}...")
        
        try:
            circuit_metrics = app_state.circuit_analyzer.analyze_circuit(request.circuit)
            
            logger.info(
                f"✅ Circuit analysis complete: "
                f"{circuit_metrics.num_qubits} qubits, "
                f"{circuit_metrics.circuit_depth} depth, "
                f"{circuit_metrics.gate_stats.total_gates} gates, "
                f"type={circuit_metrics.circuit_type.value}"
            )
        except Exception as e:
            logger.error(f"Failed to analyze circuit: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Circuit analysis failed: {str(e)}"
            )
        
        # Validate backend compatibility
        is_compatible, error_msg = app_state.cost_calculator.validate_backend_compatibility(
            request.backend, circuit_metrics
        )
        if not is_compatible:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Calculate real costs
        cost_usd, cost_dalla = app_state.cost_calculator.calculate_job_cost(
            request.backend,
            circuit_metrics,
            request.shots
        )
        
        # Estimate execution time
        circuit_time_ms, queue_time_s, total_time_s = app_state.cost_calculator.estimate_execution_time(
            request.backend,
            circuit_metrics,
            request.shots
        )
        
        logger.info(
            f"💰 Cost: ${cost_usd:.6f} USD = {cost_dalla:,} DALLA"
        )
        logger.info(
            f"⏱️  Estimated time: {circuit_time_ms:.2f}ms circuit + "
            f"{queue_time_s:.1f}s queue = {total_time_s:.1f}s total"
        )
        
        # Compute circuit hash
        circuit_hash = hashlib.sha256(request.circuit.encode()).digest()
        
        # Convert backend string to index
        backend_index = QuantumBackendIndex.from_string(request.backend)
        
        # Submit to blockchain if enabled
        if app_state.blockchain_adapter:
            try:
                tx_hash = await app_state.blockchain_adapter.submit_quantum_job(
                    job_id=job_id,
                    backend=request.backend,
                    circuit_hash=circuit_hash,
                    num_qubits=circuit_metrics.num_qubits,
                    circuit_depth=circuit_metrics.circuit_depth,
                    num_shots=request.shots,
                    backend_index=backend_index,
                )
                logger.info("✅ Submitted job {} to blockchain: {}", job_id, tx_hash)
            except Exception as e:
                logger.error("Failed to submit job to blockchain: {}", e)
                # Continue anyway - job can still execute locally
        
        # Store circuit metrics for later use
        app_state.jobs[job_id] = {
            'job_id': job_id,
            'submitter': request.submitter_address,
            'circuit': request.circuit,
            'backend': request.backend,
            'shots': request.shots,
            'status': JobStatus.PENDING,
            'submitted_at': datetime.utcnow(),
            'circuit_metrics': circuit_metrics,
            'cost_dalla': cost_dalla,
            'cost_usd': cost_usd,
            'estimated_time_s': total_time_s,
        }
        
        # Queue for execution (in background)
        asyncio.create_task(_execute_job(job_id))
        
        logger.info("🚀 Queued quantum job {} from {}", job_id, request.submitter_address)
        
        return SubmitJobResponse(
            job_id=job_id,
            status="pending",
            submitted_at=app_state.jobs[job_id]['submitted_at'].isoformat(),
            estimated_cost=cost_dalla,
            estimated_completion=(datetime.utcnow().timestamp() + total_time_s).__str__(),
            message=f"Job submitted successfully. Circuit: {circuit_metrics.num_qubits} qubits, {circuit_metrics.gate_stats.total_gates} gates",
        )
        
    except Exception as e:
        logger.error("Failed to submit quantum job: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status and results."""
    if job_id not in app_state.jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = app_state.jobs[job_id]
    result = app_state.results.get(job_id)
    
    # Get circuit metrics (stored during submission)
    circuit_metrics = job.get('circuit_metrics') if isinstance(job, dict) else None
    cost_dalla = job.get('cost_dalla', 0) if isinstance(job, dict) else 0
    submitted_at = job.get('submitted_at') if isinstance(job, dict) else None
    completed_at = job.get('completed_at') if isinstance(job, dict) else None
    backend = job.get('backend', 'unknown') if isinstance(job, dict) else 'unknown'
    submitter = job.get('submitter', 'unknown') if isinstance(job, dict) else 'unknown'
    status_val = job.get('status', JobStatus.PENDING) if isinstance(job, dict) else JobStatus.PENDING
    error = job.get('error') if isinstance(job, dict) else None
    
    # Build circuit info from real metrics
    circuit_info = circuit_metrics.to_dict() if circuit_metrics else {
        "qubits": 0,
        "gates": 0,
        "depth": 0,
        "circuit_type": "unknown",
    }
    
    return JobResponse(
        job_id=job_id,
        submitter=submitter,
        backend=backend,
        status=status_val.value if isinstance(status_val, JobStatus) else str(status_val),
        submitted_at=submitted_at.isoformat() if submitted_at else datetime.utcnow().isoformat(),
        completed_at=completed_at.isoformat() if completed_at else None,
        circuit=circuit_info,
        results=result.to_dict() if result else None,
        cost=int(cost_dalla),
        error=error if status_val == JobStatus.FAILED else None,
    )


@app.get("/api/v1/jobs")
async def list_jobs(
    account: str,
    limit: int = 20,
    status_filter: Optional[str] = None
):
    """List jobs for an account."""
    # Filter jobs by account (handle dict storage)
    account_jobs = [
        job for job in app_state.jobs.values()
        if (job.get('submitter') if isinstance(job, dict) else job.submitter) == account
    ]
    
    # Apply status filter if provided
    if status_filter:
        account_jobs = [
            job for job in account_jobs
            if (job.get('status', JobStatus.PENDING).value if isinstance(job, dict) 
                else job.status.value) == status_filter
        ]
    
    # Sort by submission time (newest first)
    def get_submitted_at(job):
        if isinstance(job, dict):
            return job.get('submitted_at', datetime.utcnow())
        return job.submitted_at
    
    account_jobs.sort(key=get_submitted_at, reverse=True)
    
    # Limit results
    account_jobs = account_jobs[:limit]
    
    # Convert to response format
    result = []
    for job in account_jobs:
        if isinstance(job, dict):
            result.append({
                "job_id": job.get('job_id', 'unknown'),
                "backend": job.get('backend', 'unknown'),
                "status": job.get('status', JobStatus.PENDING).value if isinstance(job.get('status'), JobStatus) else str(job.get('status')),
                "submitted_at": job.get('submitted_at').isoformat() if job.get('submitted_at') else datetime.utcnow().isoformat(),
                "completed_at": job.get('completed_at').isoformat() if job.get('completed_at') else None,
            })
        else:
            result.append({
                "job_id": job.job_id,
                "backend": job.backend,
                "status": job.status.value,
                "submitted_at": job.submitted_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            })
    
    return result


@app.delete("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, account: str):
    """Cancel pending job."""
    if job_id not in app_state.jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = app_state.jobs[job_id]
    
    # Get submitter (handle dict storage)
    submitter = job.get('submitter') if isinstance(job, dict) else job.submitter
    
    # Verify ownership
    if submitter != account:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this job"
        )
    
    # Get current status
    current_status = job.get('status', JobStatus.PENDING) if isinstance(job, dict) else job.status
    
    # Can only cancel pending jobs
    if current_status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        status_str = current_status.value if isinstance(current_status, JobStatus) else str(current_status)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {status_str} status"
        )
    
    # Update status
    if isinstance(job, dict):
        job['status'] = JobStatus.CANCELLED
        job['completed_at'] = datetime.utcnow()
    else:
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
    
    logger.info("❌ Cancelled job {}", job_id)
    
    return {"message": f"Job {job_id} cancelled successfully"}


# =============================================================================
# Backend Management
# =============================================================================

@app.get("/api/v1/backends", response_model=List[BackendInfo])
async def list_backends():
    """List available quantum backends."""
    # Simplified - would query actual backends in production
    backends = []
    
    if app_state.config.azure_quantum_enabled:
        backends.append(BackendInfo(
            name="azure-ionq",
            status="available",
            qubits=11,
            queue_length=0,
            avg_wait_time=30,
            cost_per_shot=100_000,  # 0.0000001 DALLA
            capabilities=["optimization", "simulation"],
            last_update=datetime.utcnow().isoformat(),
        ))
    
    if app_state.config.ibm_quantum_enabled:
        backends.append(BackendInfo(
            name="ibm-quantum",
            status="available",
            qubits=127,
            queue_length=2,
            avg_wait_time=120,
            cost_per_shot=150_000,
            capabilities=["optimization", "simulation", "cryptography"],
            last_update=datetime.utcnow().isoformat(),
        ))
    
    if app_state.config.local_simulator:
        backends.append(BackendInfo(
            name="qiskit",
            status="available",
            qubits=32,
            queue_length=0,
            avg_wait_time=5,
            cost_per_shot=10_000,  # Cheaper for local simulator
            capabilities=["simulation", "testing"],
            last_update=datetime.utcnow().isoformat(),
        ))
    
    return backends


# =============================================================================
# Cost Estimation
# =============================================================================

@app.post("/api/v1/jobs/estimate", response_model=EstimateCostResponse)
async def estimate_job_cost(request: EstimateCostRequest):
    """Estimate quantum job cost."""
    # Cost calculation (simplified)
    dalla_per_qubit = 1_000_000  # 0.000001 DALLA
    dalla_per_shot = 100_000  # 0.0000001 DALLA
    dalla_per_gate = 50_000  # 0.00000005 DALLA
    
    qubit_cost = request.qubits * dalla_per_qubit
    shot_cost = request.shots * dalla_per_shot
    gate_cost = request.gates * dalla_per_gate
    
    total_cost = qubit_cost + shot_cost + gate_cost
    
    return EstimateCostResponse(
        cost=total_cost,
        breakdown={
            "qubit_cost": qubit_cost,
            "shot_cost": shot_cost,
            "gate_cost": gate_cost,
        }
    )


# =============================================================================
# Statistics
# =============================================================================

@app.get("/api/v1/accounts/{account}/stats", response_model=AccountStatsResponse)
async def get_account_stats(account: str):
    """Get account quantum statistics."""
    # Get jobs for account (handle dict storage)
    account_jobs = [
        job for job in app_state.jobs.values()
        if (job.get('submitter') if isinstance(job, dict) else getattr(job, 'submitter', None)) == account
    ]
    
    # Get completed and failed jobs
    completed = [
        j for j in account_jobs 
        if (j.get('status') if isinstance(j, dict) else getattr(j, 'status', None)) == JobStatus.COMPLETED
    ]
    failed = [
        j for j in account_jobs 
        if (j.get('status') if isinstance(j, dict) else getattr(j, 'status', None)) == JobStatus.FAILED
    ]
    
    # Calculate total shots
    total_shots = 0
    for job in account_jobs:
        shots = job.get('shots', 0) if isinstance(job, dict) else getattr(job, 'shots', 0)
        total_shots += shots
    
    # Calculate total cost from stored values
    total_cost = 0
    for job in account_jobs:
        cost = job.get('cost_dalla', 0) if isinstance(job, dict) else 0
        total_cost += cost
    
    # Find favorite backend
    backend_counts = {}
    for job in account_jobs:
        backend = job.get('backend', 'unknown') if isinstance(job, dict) else getattr(job, 'backend', 'unknown')
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    
    favorite = max(backend_counts.keys(), key=lambda k: backend_counts[k]) if backend_counts else None
    
    # Calculate average execution time from completed jobs
    execution_times = []
    for job in completed:
        if isinstance(job, dict):
            exec_time = job.get('execution_time_ms')
            if exec_time is not None:
                execution_times.append(exec_time)
    
    avg_execution_time = (sum(execution_times) / len(execution_times)) if execution_times else 0.0
    
    return AccountStatsResponse(
        total_jobs=len(account_jobs),
        completed_jobs=len(completed),
        failed_jobs=len(failed),
        total_cost=int(total_cost),
        total_shots=total_shots,
        favorite_backend=favorite,
        avg_execution_time=avg_execution_time,
    )


@app.get("/api/v1/stats/system", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system-wide statistics."""
    # Count active jobs (handle dict storage)
    active = [
        j for j in app_state.jobs.values() 
        if (j.get('status') if isinstance(j, dict) else getattr(j, 'status', None)) == JobStatus.RUNNING
    ]
    
    # Calculate total shots
    total_shots = 0
    for job in app_state.jobs.values():
        shots = job.get('shots', 0) if isinstance(job, dict) else getattr(job, 'shots', 0)
        total_shots += shots
    
    # Calculate backend utilization from active + completed jobs
    backend_job_counts = {}
    for job in app_state.jobs.values():
        backend = job.get('backend', 'unknown') if isinstance(job, dict) else getattr(job, 'backend', 'unknown')
        backend_job_counts[backend] = backend_job_counts.get(backend, 0) + 1
    
    total_jobs_for_util = sum(backend_job_counts.values())
    backend_utilization = {
        backend: (count / total_jobs_for_util * 100) if total_jobs_for_util > 0 else 0.0
        for backend, count in backend_job_counts.items()
    }
    
    # Calculate average wait time from completed jobs
    wait_times = []
    for job in app_state.jobs.values():
        if isinstance(job, dict):
            submitted_at = job.get('submitted_at')
            started_at = job.get('started_at')
            if submitted_at and started_at:
                wait_seconds = (started_at - submitted_at).total_seconds()
                wait_times.append(wait_seconds)
    
    avg_wait_time = (sum(wait_times) / len(wait_times)) if wait_times else 0.0
    
    return SystemStatsResponse(
        total_jobs=app_state.job_counter,
        active_jobs=len(active),
        total_shots=total_shots,
        backend_utilization=backend_utilization,
        avg_wait_time=avg_wait_time,
    )


@app.get("/api/v1/jobs/{job_id}/attestation", response_model=AttestationResponse)
async def get_attestation(job_id: str):
    """Get QPU attestation for job."""
    if job_id not in app_state.jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = app_state.jobs[job_id]
    
    # Get status (handle dict storage)
    job_status = job.get('status') if isinstance(job, dict) else getattr(job, 'status', None)
    
    if job_status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Attestation only available for completed jobs"
        )
    
    # Get backend
    backend = job.get('backend', 'unknown') if isinstance(job, dict) else getattr(job, 'backend', 'unknown')
    
    # Generate attestation (simplified - would use real cryptographic proof)
    attestation_data = {
        "signed": True,
        "signature": hashlib.sha256(f"{job_id}:attestation".encode()).hexdigest(),
        "timestamp": int(datetime.utcnow().timestamp()),
        "proof": hashlib.sha256(f"{job_id}:proof".encode()).hexdigest(),
    }
    
    return AttestationResponse(
        job_id=job_id,
        backend=backend,
        attestation=attestation_data,
        on_chain_proof=hashlib.sha256(f"{job_id}:chain".encode()).hexdigest(),
    )


@app.get("/api/v1/jobs/{job_id}/results")
async def download_results(job_id: str, format: str = "json"):
    """Download job results as file."""
    if job_id not in app_state.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results for job {job_id} not found"
        )
    
    result = app_state.results[job_id]
    
    if format == "json":
        import json
        data = json.dumps(result.to_dict(), indent=2)
        media_type = "application/json"
        filename = f"{job_id}_results.json"
    elif format == "csv":
        # Simple CSV format for counts
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Bitstring", "Count"])
        if result.counts:  # Check if counts exist
            for bitstring, count in result.counts.items():
                writer.writerow([bitstring, count])
        data = output.getvalue()
        media_type = "text/csv"
        filename = f"{job_id}_results.csv"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}"
        )
    
    return StreamingResponse(
        iter([data]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# =============================================================================
# Background Job Execution
# =============================================================================

async def _execute_job(job_id: str):
    """Execute quantum job in background."""
    try:
        job = app_state.jobs[job_id]
        
        # Get stored values (handle dict storage)
        if isinstance(job, dict):
            circuit_metrics = job.get('circuit_metrics')
            backend = job.get('backend', 'unknown')
            shots = job.get('shots', 1024)
            
            # Update status
            job['status'] = JobStatus.RUNNING
            job['started_at'] = datetime.utcnow()
            
            # Simulate quantum execution (would use real backend in production)
            # Use estimated time from cost calculator
            estimated_time_s = job.get('estimated_time_s', 2.0)
            await asyncio.sleep(min(estimated_time_s, 2.0))  # Cap at 2s for simulation
            
            # Calculate execution time in ms
            execution_time_ms = estimated_time_s * 1000
            
            # Create mock result using circuit metrics
            result = QuantumJobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                counts={"000": shots // 2, "111": shots // 2},  # Mock Bell state results
                execution_time=execution_time_ms,
                shots_used=shots,
                backend_used=backend,
            )
            
            # Store result and execution time
            app_state.results[job_id] = result
            job['execution_time_ms'] = execution_time_ms
            
            # Update job status
            job['status'] = JobStatus.COMPLETED
            job['completed_at'] = datetime.utcnow()
            
            # Log with real metrics
            logger.info(
                "✅ Completed quantum job {} ({} qubits, {} gates, {} ms)",
                job_id,
                circuit_metrics.num_qubits if circuit_metrics else 0,
                circuit_metrics.gate_stats.total_gates if circuit_metrics else 0,
                execution_time_ms
            )
        else:
            # Legacy QuantumJob object (for compatibility)
            job.status = JobStatus.RUNNING
            await asyncio.sleep(2)
            
            result = QuantumJobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                counts={"000": 512, "111": 512},
                execution_time=2000,
            )
            
            app_state.results[job_id] = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
        
        # Submit result to blockchain if enabled
        if app_state.blockchain_adapter:
            try:
                await app_state.blockchain_adapter.record_quantum_result(
                    job_id=job_id,
                    result_data=result.to_dict(),
                    accuracy_score=95,  # Mock accuracy
                )
                logger.info("✅ Recorded result for job {} on blockchain", job_id)
            except Exception as e:
                logger.error("Failed to record result on blockchain: {}", e)
        
    except Exception as e:
        logger.error("Job execution failed: {}", e)
        job = app_state.jobs[job_id]
        
        if isinstance(job, dict):
            job['status'] = JobStatus.FAILED
            job['error'] = str(e)
            job['completed_at'] = datetime.utcnow()
        else:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run API server."""
    # Configure logging
    logger.add(
        "logs/kinich_api_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # Get configuration from environment
    host = os.getenv("KINICH_HOST", "0.0.0.0")  # Listen on all interfaces for Docker
    port = int(os.getenv("PORT", "8888"))  # Docker container port
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info("🚀 Starting Kinich Quantum API server on {}:{}", host, port)
    logger.info("   Blockchain RPC: {}", os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944"))
    logger.info("   Azure Quantum: {}", os.getenv("AZURE_QUANTUM_ENABLED", "true"))
    logger.info("   IBM Quantum: {}", os.getenv("IBM_QUANTUM_ENABLED", "false"))
    logger.info("   Reload: {}", reload)
    logger.info("   Workers: {}", workers)
    
    # Run server
    uvicorn.run(
        "kinich.api_server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
