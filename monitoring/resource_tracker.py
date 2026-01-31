"""
Azure Quantum credit tracking
Monitor and account for quantum resource usage per user
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class QuantumResourceTracker:
    """
    Track Azure Quantum credit usage per account
    Enables cost allocation and billing
    """
    
    def __init__(self):
        self.usage_by_user: Dict[str, float] = {}
    
    def track_job(self, user_id: str, backend: str, shots: int, qubits: int):
        """
        Record quantum job resource usage
        
        Args:
            user_id: User or BelizeID
            backend: Quantum backend used
            shots: Number of circuit executions
            qubits: Number of qubits
        """
        # Simplified cost calculation (1 eHQC = 1 Azure Quantum credit)
        cost = self._calculate_cost(backend, shots, qubits)
        
        if user_id not in self.usage_by_user:
            self.usage_by_user[user_id] = 0.0
        
        self.usage_by_user[user_id] += cost
        logger.info(f"💰 User {user_id} used {cost:.2f} eHQC (total: {self.usage_by_user[user_id]:.2f})")
    
    def _calculate_cost(self, backend: str, shots: int, qubits: int) -> float:
        """Calculate Azure Quantum cost in eHQC credits"""
        if backend == "simulator":
            return 0.0  # Simulators are free
        elif backend == "azure":
            # IonQ pricing: ~$0.00003 per shot per qubit
            return shots * qubits * 0.00003
        elif backend == "ibm":
            # IBM Quantum pricing varies
            return shots * qubits * 0.00001
        return 0.0
    
    def get_user_usage(self, user_id: str) -> float:
        """Get total usage for a user"""
        return self.usage_by_user.get(user_id, 0.0)
