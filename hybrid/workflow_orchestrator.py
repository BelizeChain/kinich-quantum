"""
Hybrid classical-quantum workflow orchestrator
Automatically decomposes large problems into quantum+classical steps
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class HybridWorkflow:
    """
    Orchestrate hybrid classical-quantum computations
    
    Decomposes problems too large for quantum hardware into:
    1. Classical preprocessing
    2. Quantum subroutines (within hardware limits)
    3. Classical postprocessing
    """
    
    def __init__(self, max_quantum_qubits: int = 20):
        self.max_quantum_qubits = max_quantum_qubits
    
    async def decompose_problem(self, problem: Dict[str, Any]) -> List[Dict]:
        """
        Decompose large problem into hybrid steps
        
        Returns:
            List of workflow steps (classical or quantum)
        """
        steps = []
        
        problem_size = problem.get("size", 0)
        
        if problem_size <= self.max_quantum_qubits:
            # Fits on quantum hardware - single quantum step
            steps.append({"type": "quantum", "problem": problem})
        else:
            # Too large - decompose into chunks
            steps.append({"type": "classical", "task": "preprocess", "problem": problem})
            
            # Split into quantum-sized chunks
            num_chunks = (problem_size + self.max_quantum_qubits - 1) // self.max_quantum_qubits
            for i in range(num_chunks):
                steps.append({
                    "type": "quantum",
                    "task": f"chunk_{i}",
                    "qubits": min(self.max_quantum_qubits, problem_size - i * self.max_quantum_qubits)
                })
            
            steps.append({"type": "classical", "task": "aggregate", "num_chunks": num_chunks})
        
        logger.info(f"Decomposed problem into {len(steps)} steps")
        return steps
