"""
Job Scheduler for Kinich

Intelligent job scheduling and distribution across multiple
quantum backends with load balancing and retry logic.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import heapq

from . import (
    QuantumJob,
    JobStatus,
    JobPriority,
    JobType,
)

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Job scheduling strategies."""
    
    PRIORITY_FIRST = "priority_first"  # Highest priority first
    FIFO = "fifo"  # First-in-first-out
    SHORTEST_JOB_FIRST = "shortest_job_first"  # Estimated shortest runtime
    COST_OPTIMIZED = "cost_optimized"  # Minimize execution cost
    LOAD_BALANCED = "load_balanced"  # Balance across backends


@dataclass
class SchedulerConfig:
    """Configuration for job scheduler."""
    
    # Scheduling strategy
    strategy: SchedulingStrategy = field(default=SchedulingStrategy.PRIORITY_FIRST)
    
    # Queue settings
    max_queue_size: int = field(default=1000)
    enable_priority_queues: bool = field(default=True)
    
    # Retry settings
    max_retries: int = field(default=3)
    retry_delay: int = field(default=60)  # seconds
    exponential_backoff: bool = field(default=True)
    
    # Timeout settings
    default_timeout: int = field(default=600)  # seconds
    timeout_by_job_type: Dict[JobType, int] = field(default_factory=dict)
    
    # Load balancing
    enable_load_balancing: bool = field(default=True)
    max_jobs_per_backend: int = field(default=10)
    
    # Cost optimization
    max_cost_per_job: Optional[float] = field(default=None)
    prefer_free_backends: bool = field(default=True)


class JobScheduler:
    """
    Intelligent job scheduler for distributed quantum computing.
    
    Responsibilities:
    - Queue management with priority
    - Backend selection and load balancing
    - Retry logic for failed jobs
    - Timeout handling
    - Cost optimization
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        adapter_registry=None
    ):
        """
        Initialize job scheduler.
        
        Args:
            config: Scheduler configuration
            adapter_registry: Adapter registry for backend selection
        """
        self.config = config or SchedulerConfig()
        self.adapter_registry = adapter_registry
        
        # Priority queues (one per priority level)
        self._priority_queues: Dict[JobPriority, List] = {
            JobPriority.CRITICAL: [],
            JobPriority.HIGH: [],
            JobPriority.NORMAL: [],
            JobPriority.LOW: [],
            JobPriority.BATCH: [],
        }
        
        # Job tracking
        self._pending_jobs: Dict[str, QuantumJob] = {}
        self._running_jobs: Dict[str, QuantumJob] = {}
        self._completed_jobs: Dict[str, QuantumJob] = {}
        self._failed_jobs: Dict[str, QuantumJob] = {}
        
        # Backend load tracking
        self._backend_loads: Dict[str, int] = {}
        
        # Statistics
        self._total_scheduled = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_retries = 0
        
        # Callbacks
        self._on_job_scheduled: List[Callable] = []
        self._on_job_completed: List[Callable] = []
        self._on_job_failed: List[Callable] = []
        
        logger.info(f"Initialized job scheduler (strategy: {self.config.strategy.value})")
    
    def schedule_job(self, job: QuantumJob) -> bool:
        """
        Schedule a quantum job for execution.
        
        Args:
            job: Quantum job to schedule
        
        Returns:
            True if scheduled, False if rejected
        """
        job_id = job.get_job_id()
        
        # Check queue size
        total_queued = sum(len(q) for q in self._priority_queues.values())
        if total_queued >= self.config.max_queue_size:
            logger.warning(f"Queue full, rejecting job {job_id}")
            return False
        
        # Add to appropriate priority queue
        priority = job.priority
        queue = self._priority_queues[priority]
        
        # Create heap entry (priority value, timestamp, job)
        entry = (priority.value, datetime.now(), job)
        heapq.heappush(queue, entry)
        
        self._pending_jobs[job_id] = job
        self._total_scheduled += 1
        
        logger.info(
            f"Scheduled job {job_id} "
            f"(type: {job.job_type.name}, priority: {priority.name})"
        )
        
        # Trigger callbacks
        for callback in self._on_job_scheduled:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return True
    
    async def get_next_job(self) -> Optional[QuantumJob]:
        """
        Get next job to execute based on scheduling strategy.
        
        Returns:
            Next job or None if queue is empty
        """
        if self.config.strategy == SchedulingStrategy.PRIORITY_FIRST:
            return await self._get_next_by_priority()
        
        elif self.config.strategy == SchedulingStrategy.FIFO:
            return await self._get_next_fifo()
        
        elif self.config.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            return await self._get_next_shortest()
        
        elif self.config.strategy == SchedulingStrategy.COST_OPTIMIZED:
            return await self._get_next_cost_optimized()
        
        elif self.config.strategy == SchedulingStrategy.LOAD_BALANCED:
            return await self._get_next_load_balanced()
        
        return None
    
    async def _get_next_by_priority(self) -> Optional[QuantumJob]:
        """Get highest priority job."""
        # Check queues in priority order
        for priority in JobPriority:
            queue = self._priority_queues[priority]
            
            if queue:
                _, _, job = heapq.heappop(queue)
                return job
        
        return None
    
    async def _get_next_fifo(self) -> Optional[QuantumJob]:
        """Get oldest job regardless of priority."""
        oldest_job = None
        oldest_time = None
        oldest_queue = None
        
        for priority, queue in self._priority_queues.items():
            if queue:
                _, timestamp, job = queue[0]  # Peek at first item
                
                if oldest_time is None or timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_job = job
                    oldest_queue = queue
        
        if oldest_queue:
            heapq.heappop(oldest_queue)
        
        return oldest_job
    
    async def _get_next_shortest(self) -> Optional[QuantumJob]:
        """Get job with shortest estimated runtime."""
        shortest_job = None
        shortest_time = None
        shortest_queue = None
        
        for priority, queue in self._priority_queues.items():
            for _, _, job in queue:
                estimated = getattr(job, 'estimated_execution_time', 300)
                
                if shortest_time is None or estimated < shortest_time:
                    shortest_time = estimated
                    shortest_job = job
                    shortest_queue = queue
        
        if shortest_queue:
            # Remove from queue
            shortest_queue[:] = [
                item for item in shortest_queue
                if item[2].get_job_id() != shortest_job.get_job_id()
            ]
            heapq.heapify(shortest_queue)
        
        return shortest_job
    
    async def _get_next_cost_optimized(self) -> Optional[QuantumJob]:
        """Get job that can be executed at lowest cost."""
        # For now, prefer jobs that can run on free simulators
        return await self._get_next_by_priority()
    
    async def _get_next_load_balanced(self) -> Optional[QuantumJob]:
        """Get job for least loaded backend."""
        # Select job based on backend availability
        return await self._get_next_by_priority()
    
    def select_backend_for_job(self, job: QuantumJob) -> Optional[str]:
        """
        Select best backend for job execution.
        
        Args:
            job: Quantum job
        
        Returns:
            Backend name or None
        """
        if self.adapter_registry is None:
            return None
        
        # Get qubit requirement
        min_qubits = getattr(job, 'estimated_qubits', 2)
        
        # Check backend preference
        if job.backend_preference:
            for backend_name in job.backend_preference:
                # Check if backend is available and not overloaded
                load = self._backend_loads.get(backend_name, 0)
                
                if load < self.config.max_jobs_per_backend:
                    return backend_name
        
        # Auto-select based on requirements
        backend = self.adapter_registry.find_suitable_adapter(
            min_qubits=min_qubits,
            prefer_simulator=self.config.prefer_free_backends,
            max_cost=self.config.max_cost_per_job
        )
        
        return backend
    
    def mark_job_running(self, job: QuantumJob, backend: str) -> None:
        """Mark job as running on backend."""
        job_id = job.get_job_id()
        
        if job_id in self._pending_jobs:
            del self._pending_jobs[job_id]
        
        self._running_jobs[job_id] = job
        
        # Update backend load
        self._backend_loads[backend] = self._backend_loads.get(backend, 0) + 1
        
        logger.info(f"Job {job_id} running on {backend}")
    
    def mark_job_completed(self, job: QuantumJob, backend: str) -> None:
        """Mark job as completed."""
        job_id = job.get_job_id()
        
        if job_id in self._running_jobs:
            del self._running_jobs[job_id]
        
        self._completed_jobs[job_id] = job
        self._total_completed += 1
        
        # Update backend load
        if backend in self._backend_loads:
            self._backend_loads[backend] = max(0, self._backend_loads[backend] - 1)
        
        logger.info(f"Job {job_id} completed")
        
        # Trigger callbacks
        for callback in self._on_job_completed:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def mark_job_failed(self, job: QuantumJob, backend: str, error: str) -> bool:
        """
        Mark job as failed and potentially retry.
        
        Args:
            job: Failed job
            backend: Backend where it failed
            error: Error message
        
        Returns:
            True if job will be retried, False otherwise
        """
        job_id = job.get_job_id()
        
        if job_id in self._running_jobs:
            del self._running_jobs[job_id]
        
        # Update backend load
        if backend in self._backend_loads:
            self._backend_loads[backend] = max(0, self._backend_loads[backend] - 1)
        
        # Check if we should retry
        if job.can_retry():
            logger.warning(
                f"Job {job_id} failed (attempt {job.attempts}), will retry: {error}"
            )
            
            # Calculate retry delay
            delay = self.config.retry_delay
            if self.config.exponential_backoff:
                delay *= (2 ** (job.attempts - 1))
            
            # Re-schedule after delay
            asyncio.create_task(self._retry_job_after_delay(job, delay))
            self._total_retries += 1
            
            return True
        
        else:
            logger.error(
                f"Job {job_id} failed permanently after {job.attempts} attempts: {error}"
            )
            
            self._failed_jobs[job_id] = job
            self._total_failed += 1
            
            # Trigger callbacks
            for callback in self._on_job_failed:
                try:
                    callback(job, error)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            return False
    
    async def _retry_job_after_delay(self, job: QuantumJob, delay: int) -> None:
        """Retry job after delay."""
        await asyncio.sleep(delay)
        
        logger.info(f"Retrying job {job.get_job_id()}")
        self.schedule_job(job)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'pending': len(self._pending_jobs),
            'running': len(self._running_jobs),
            'completed': len(self._completed_jobs),
            'failed': len(self._failed_jobs),
            'by_priority': {
                priority.name: len(queue)
                for priority, queue in self._priority_queues.items()
            },
            'backend_loads': self._backend_loads.copy(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_processed = self._total_completed + self._total_failed
        
        return {
            'total_scheduled': self._total_scheduled,
            'total_completed': self._total_completed,
            'total_failed': self._total_failed,
            'total_retries': self._total_retries,
            'success_rate': (
                self._total_completed / total_processed
                if total_processed > 0 else 0.0
            ),
            'retry_rate': (
                self._total_retries / self._total_scheduled
                if self._total_scheduled > 0 else 0.0
            ),
            'queue_status': self.get_queue_status(),
        }
    
    def on_job_scheduled(self, callback: Callable) -> None:
        """Register callback for job scheduled."""
        self._on_job_scheduled.append(callback)
    
    def on_job_completed(self, callback: Callable) -> None:
        """Register callback for job completed."""
        self._on_job_completed.append(callback)
    
    def on_job_failed(self, callback: Callable) -> None:
        """Register callback for job failed."""
        self._on_job_failed.append(callback)
