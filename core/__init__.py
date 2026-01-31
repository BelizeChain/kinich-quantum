"""
Kinich Core Module

Core quantum job types, status enums, and job classes for
distributed quantum computing in BelizeChain.

Author: BelizeChain Team
License: MIT
"""

# Import job types first (no dependencies)
from .jobs import (
    JobType,
    JobStatus,
    JobPriority,
    QuantumJob,
    QuantumJobMetadata,
    QuantumJobResult,
    CryptographyJob,
    OptimizationJob,
    SimulationJob,
    AIEnhancementJob,
    ConsensusJob,
)

# Then import components that depend on job types
from .circuit_optimizer import CircuitOptimizer, OptimizationConfig, OptimizationLevel
from .result_aggregator import ResultAggregator, AggregatedResult
from .quantum_node import QuantumNode, NodeConfig, NodeStatus
from .job_scheduler import JobScheduler, SchedulerConfig, SchedulingStrategy

__all__ = [
    # Enums
    "JobType",
    "JobStatus",
    "JobPriority",
    # Base classes
    "QuantumJob",
    "QuantumJobMetadata",
    "QuantumJobResult",
    # Specialized jobs
    "CryptographyJob",
    "OptimizationJob",
    "SimulationJob",
    "AIEnhancementJob",
    "ConsensusJob",
    # Node components
    "QuantumNode",
    "NodeConfig",
    "NodeStatus",
    # Scheduling
    "JobScheduler",
    "SchedulerConfig",
    "SchedulingStrategy",
    # Optimization
    "CircuitOptimizer",
    "OptimizationConfig",
    "OptimizationLevel",
    # Results
    "ResultAggregator",
    "AggregatedResult",
]
