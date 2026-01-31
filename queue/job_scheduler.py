"""
Quantum Job Queue System using Redis
Priority-based scheduling for quantum workloads
"""

import asyncio
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis[async] not installed. Run: pip install redis[async]")

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class JobStatus(Enum):
    """Job lifecycle states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuantumJob:
    """Quantum computation job"""
    job_id: str
    circuit_qasm: str
    backend: str  # "azure", "ibm", "simulator"
    priority: JobPriority
    user_id: str
    created_at: str
    circuit_depth: int
    num_qubits: int
    shots: int = 1024
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None


class QuantumJobQueue:
    """
    Redis-backed job queue for quantum workloads
    Features:
    - Priority-based scheduling
    - Persistent job state
    - Job cancellation
    - Dead-letter queue for failed jobs
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required. Install with: pip install redis[async]")
        
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        
        # Redis key prefixes
        self.QUEUE_PREFIX = "kinich:queue:"
        self.JOB_PREFIX = "kinich:job:"
        self.RUNNING_SET = "kinich:running"
        self.DLQ_PREFIX = "kinich:dlq:"  # Dead letter queue
    
    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        logger.info(f"✅ Connected to Redis: {self.redis_url}")
    
    async def enqueue(self, job: QuantumJob) -> str:
        """
        Add job to queue with priority
        
        Returns:
            job_id
        """
        if self.redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")
        
        # Store job data
        job_key = f"{self.JOB_PREFIX}{job.job_id}"
        await self.redis.set(job_key, json.dumps(asdict(job)))
        
        # Add to priority queue (sorted set with priority as score)
        queue_key = f"{self.QUEUE_PREFIX}{job.backend}"
        await self.redis.zadd(queue_key, {job.job_id: job.priority.value})
        
        logger.info(f"📥 Enqueued job {job.job_id} (priority={job.priority.name}, backend={job.backend})")
        return job.job_id
    
    async def dequeue(self, backend: str, timeout: int = 30) -> Optional[QuantumJob]:
        """
        Get highest-priority job from queue (blocking with timeout)
        
        Args:
            backend: Backend name ("azure", "ibm", "simulator")
            timeout: Block for up to N seconds if queue empty
        
        Returns:
            QuantumJob or None if timeout
        """
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        queue_key = f"{self.QUEUE_PREFIX}{backend}"
        
        # BZPOPMAX: blocking pop with highest score (priority)
        result = await self.redis.bzpopmax(queue_key, timeout)
        
        if result is None:
            return None
        
        _, job_id, _ = result
        
        # Fetch job data
        job_key = f"{self.JOB_PREFIX}{job_id}"
        job_data = await self.redis.get(job_key)
        
        if job_data is None:
            logger.error(f"Job {job_id} not found in storage")
            return None
        
        job_dict = json.loads(job_data)
        job_dict['priority'] = JobPriority(job_dict['priority'])
        job_dict['status'] = JobStatus(job_dict['status'])
        job = QuantumJob(**job_dict)
        
        # Mark as running
        job.status = JobStatus.RUNNING
        await self.update_job(job)
        await self.redis.sadd(self.RUNNING_SET, job_id)
        
        logger.info(f"📤 Dequeued job {job_id} from {backend}")
        return job
    
    async def update_job(self, job: QuantumJob):
        """Update job state"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        job_key = f"{self.JOB_PREFIX}{job.job_id}"
        await self.redis.set(job_key, json.dumps(asdict(job)))
    
    async def complete_job(self, job_id: str, result: Dict):
        """Mark job as completed with result"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        job = await self.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found")
            return
        
        job.status = JobStatus.COMPLETED
        job.result = result
        await self.update_job(job)
        
        # Remove from running set
        await self.redis.srem(self.RUNNING_SET, job_id)
        
        logger.info(f"✅ Job {job_id} completed")
    
    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed and move to DLQ"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        job = await self.get_job(job_id)
        if job is None:
            return
        
        job.status = JobStatus.FAILED
        job.error = error
        await self.update_job(job)
        
        # Move to dead-letter queue
        dlq_key = f"{self.DLQ_PREFIX}{job.backend}"
        await self.redis.zadd(dlq_key, {job_id: datetime.utcnow().timestamp()})
        
        # Remove from running
        await self.redis.srem(self.RUNNING_SET, job_id)
        
        logger.error(f"❌ Job {job_id} failed: {error}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        job = await self.get_job(job_id)
        if job is None or job.status == JobStatus.COMPLETED:
            return False
        
        job.status = JobStatus.CANCELLED
        await self.update_job(job)
        
        # Remove from queue if pending
        for backend in ["azure", "ibm", "simulator"]:
            queue_key = f"{self.QUEUE_PREFIX}{backend}"
            await self.redis.zrem(queue_key, job_id)
        
        # Remove from running set
        await self.redis.srem(self.RUNNING_SET, job_id)
        
        logger.info(f"🚫 Job {job_id} cancelled")
        return True
    
    async def get_job(self, job_id: str) -> Optional[QuantumJob]:
        """Retrieve job by ID"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        job_key = f"{self.JOB_PREFIX}{job_id}"
        job_data = await self.redis.get(job_key)
        
        if job_data is None:
            return None
        
        job_dict = json.loads(job_data)
        job_dict['priority'] = JobPriority(job_dict['priority'])
        job_dict['status'] = JobStatus(job_dict['status'])
        return QuantumJob(**job_dict)
    
    async def get_queue_length(self, backend: str) -> int:
        """Get number of pending jobs for a backend"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        queue_key = f"{self.QUEUE_PREFIX}{backend}"
        return await self.redis.zcard(queue_key)
    
    async def get_running_jobs(self) -> List[str]:
        """Get list of currently running job IDs"""
        if self.redis is None:
            raise RuntimeError("Not connected")
        
        return list(await self.redis.smembers(self.RUNNING_SET))
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()


# Example usage
if __name__ == "__main__":
    async def main():
        queue = QuantumJobQueue()
        await queue.connect()
        
        # Create test job
        job = QuantumJob(
            job_id="test-job-001",
            circuit_qasm="OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];",
            backend="azure",
            priority=JobPriority.HIGH,
            user_id="test-user",
            created_at=datetime.utcnow().isoformat(),
            circuit_depth=2,
            num_qubits=2
        )
        
        # Enqueue
        await queue.enqueue(job)
        
        # Get queue length
        length = await queue.get_queue_length("azure")
        print(f"Queue length: {length}")
        
        # Dequeue
        dequeued = await queue.dequeue("azure", timeout=5)
        if dequeued:
            print(f"Dequeued: {dequeued.job_id}")
            
            # Complete job
            await queue.complete_job(dequeued.job_id, {"counts": {"00": 512, "11": 512}})
        
        await queue.close()
    
    asyncio.run(main())
