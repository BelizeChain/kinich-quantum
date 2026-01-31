"""
Kinich Community Integration

Connects Kinich quantum compute orchestration to BelizeChain Community pallet
for Social Responsibility Score (SRS) tracking of quantum computing contributions.

NOTE: This module requires the separate 'nawal' package for Community pallet integration.
Install with: pip install nawal

Author: BelizeChain Quantum Team
Date: January 2026
Python: 3.13+
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from loguru import logger

# Import Community connector (requires nawal package)
try:
    from nawal.blockchain.community_connector import CommunityConnector, SRSInfo
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False
    logger.warning("CommunityConnector not available (install 'nawal' package), SRS tracking disabled for quantum jobs")


class QuantumCommunityTracker:
    """
    Tracks quantum compute contributions to Community pallet SRS system.
    
    Records quantum job completions as community participation activities,
    contributing to the participant's Social Responsibility Score.
    """
    
    def __init__(
        self,
        websocket_url: str = "ws://127.0.0.1:9944",
        mock_mode: bool = False,
        enabled: bool = True
    ):
        """
        Initialize quantum community tracker.
        
        Args:
            websocket_url: BelizeChain node WebSocket endpoint
            mock_mode: Use mock mode for testing
            enabled: Enable SRS tracking (default: True)
        """
        self.enabled = enabled and COMMUNITY_AVAILABLE
        self.community_connector: CommunityConnector | None = None
        
        if self.enabled:
            try:
                self.community_connector = CommunityConnector(
                    websocket_url=websocket_url,
                    mock_mode=mock_mode
                )
                logger.info("Quantum Community SRS tracking ENABLED")
            except Exception as e:
                logger.error(f"Failed to initialize CommunityConnector: {e}")
                self.enabled = False
        else:
            logger.info("Quantum Community SRS tracking DISABLED")
    
    async def connect(self) -> bool:
        """Connect to blockchain."""
        if not self.enabled or not self.community_connector:
            return False
        
        try:
            return await self.community_connector.connect()
        except Exception as e:
            logger.error(f"Failed to connect community connector: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from blockchain."""
        if self.community_connector:
            await self.community_connector.disconnect()
    
    async def record_quantum_job_completion(
        self,
        account_id: str,
        job_id: str,
        backend_name: str,
        shots: int,
        circuit_depth: int,
        success: bool,
        execution_time_seconds: float,
        error_mitigation_used: bool = False
    ) -> tuple[bool, str]:
        """
        Record quantum job completion for SRS tracking.
        
        Args:
            account_id: Quantum job submitter account
            job_id: Quantum job ID
            backend_name: Quantum backend used (e.g., 'ionq', 'rigetti')
            shots: Number of shots executed
            circuit_depth: Circuit depth
            success: Whether job completed successfully
            execution_time_seconds: Job execution time
            error_mitigation_used: Whether error mitigation was applied
        
        Returns:
            (success: bool, tx_hash: str)
        """
        if not self.enabled or not self.community_connector:
            return (False, "SRS tracking disabled")
        
        # Calculate quality score based on job characteristics
        quality_score = self._calculate_quantum_quality_score(
            shots=shots,
            circuit_depth=circuit_depth,
            success=success,
            error_mitigation_used=error_mitigation_used
        )
        
        # Build metadata
        metadata = {
            'job_id': job_id,
            'backend': backend_name,
            'shots': shots,
            'circuit_depth': circuit_depth,
            'execution_time': execution_time_seconds,
            'error_mitigation': error_mitigation_used,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            success, tx_hash = await self.community_connector.record_participation(
                account_id=account_id,
                activity_type='QuantumCompute',
                quality_score=quality_score,
                metadata=metadata
            )
            
            if success:
                logger.info(
                    f"Quantum job SRS recorded",
                    account=account_id[:8],
                    job_id=job_id,
                    quality=quality_score,
                    tx=tx_hash[:16]
                )
            else:
                logger.warning(f"Failed to record quantum job SRS for {account_id[:8]}")
            
            return (success, tx_hash)
        
        except Exception as e:
            logger.error(f"Quantum SRS tracking error: {e}")
            return (False, "")
    
    async def record_optimization_contribution(
        self,
        account_id: str,
        problem_type: str,  # 'VQE', 'QAOA', 'Optimization', etc.
        problem_size: int,
        iterations: int,
        convergence_achieved: bool,
        final_energy: float | None = None
    ) -> tuple[bool, str]:
        """
        Record quantum optimization problem contribution.
        
        Args:
            account_id: Problem solver account
            problem_type: Type of quantum algorithm used
            problem_size: Size of problem (qubits, variables, etc.)
            iterations: Number of iterations performed
            convergence_achieved: Whether solution converged
            final_energy: Final energy value (for VQE/QAOA)
        
        Returns:
            (success: bool, tx_hash: str)
        """
        if not self.enabled or not self.community_connector:
            return (False, "SRS tracking disabled")
        
        # Calculate quality score
        quality_score = 50.0  # Base score
        if convergence_achieved:
            quality_score += 30.0
        if problem_size >= 10:
            quality_score += 20.0  # Bonus for larger problems
        
        quality_score = min(quality_score, 100.0)
        
        # Build metadata
        metadata = {
            'problem_type': problem_type,
            'problem_size': problem_size,
            'iterations': iterations,
            'converged': convergence_achieved,
            'final_energy': final_energy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            return await self.community_connector.record_participation(
                account_id=account_id,
                activity_type='QuantumOptimization',
                quality_score=quality_score,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Optimization SRS tracking error: {e}")
            return (False, "")
    
    async def get_srs_info(self, account_id: str) -> SRSInfo | None:
        """Get SRS info for an account."""
        if not self.enabled or not self.community_connector:
            return None
        
        try:
            return await self.community_connector.get_srs_info(account_id)
        except Exception as e:
            logger.error(f"Failed to get SRS info: {e}")
            return None
    
    def _calculate_quantum_quality_score(
        self,
        shots: int,
        circuit_depth: int,
        success: bool,
        error_mitigation_used: bool
    ) -> float:
        """
        Calculate quality score for quantum job (0-100).
        
        Scoring factors:
        - Job success: 50 points
        - Circuit complexity (depth): up to 20 points
        - Shot count: up to 15 points
        - Error mitigation: 15 points bonus
        """
        score = 0.0
        
        # Success is critical
        if success:
            score += 50.0
        else:
            score += 10.0  # Partial credit for attempt
        
        # Circuit depth (normalized to typical depth range 10-100)
        depth_score = min((circuit_depth / 100.0) * 20.0, 20.0)
        score += depth_score
        
        # Shot count (normalized to typical range 100-10000)
        shot_score = min((shots / 10000.0) * 15.0, 15.0)
        score += shot_score
        
        # Error mitigation bonus
        if error_mitigation_used:
            score += 15.0
        
        return min(score, 100.0)


# =============================================================================
# Usage Example
# =============================================================================


async def example_usage():
    """Example usage of QuantumCommunityTracker."""
    
    # Initialize tracker
    tracker = QuantumCommunityTracker(
        websocket_url="ws://127.0.0.1:9944",
        mock_mode=True
    )
    
    # Connect
    await tracker.connect()
    
    # Record quantum job completion
    account = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    success, tx_hash = await tracker.record_quantum_job_completion(
        account_id=account,
        job_id="kinich_job_12345",
        backend_name="ionq",
        shots=1000,
        circuit_depth=25,
        success=True,
        execution_time_seconds=12.5,
        error_mitigation_used=True
    )
    
    print(f"Quantum job SRS recorded: {success} (tx: {tx_hash})")
    
    # Get SRS info
    srs_info = await tracker.get_srs_info(account)
    if srs_info:
        print(f"SRS Score: {srs_info.score}")
        print(f"Tier: {await tracker.community_connector.get_tier_name(srs_info.tier)}")
    
    # Disconnect
    await tracker.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
