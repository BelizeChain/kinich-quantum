"""
Quantum Contribution Tracker Service
Automatically monitors quantum job completions and submits contributions to the staking pallet.

Author: BelizeChain Team
"""

import asyncio
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from substrateinterface import SubstrateInterface, Keypair
from .belizechain_adapter import BelizeChainAdapter, JobStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContributionRecord:
    """Record of a submitted quantum contribution."""
    job_id: str
    validator_address: str
    num_qubits: int
    circuit_depth: int
    num_shots: int
    accuracy_score: int
    submitted_at: datetime
    tx_hash: str


class QuantumContributionTracker:
    """
    Monitors quantum job completions and automatically submits contributions
    to the BelizeChain staking pallet for validator rewards.
    """

    def __init__(
        self,
        blockchain_url: str = "ws://127.0.0.1:9944",
        validator_keypair: Optional[Keypair] = None,
        check_interval: int = 10,
    ):
        """
        Initialize the contribution tracker.

        Args:
            blockchain_url: WebSocket URL of BelizeChain node
            validator_keypair: Keypair of the validator (for signing transactions)
            check_interval: Seconds between job checks
        """
        self.blockchain_url = blockchain_url
        self.validator_keypair = validator_keypair or Keypair.create_from_uri('//Alice')
        self.check_interval = check_interval
        
        self.adapter = BelizeChainAdapter(blockchain_url)
        self.substrate: Optional[SubstrateInterface] = None
        
        # Track processed jobs to avoid duplicates
        self.processed_jobs: set = set()
        self.contribution_history: List[ContributionRecord] = []
        
        # Statistics
        self.total_contributions = 0
        self.total_qubits_processed = 0
        self.total_shots_executed = 0
        
        self.running = False

    async def start(self):
        """Start the contribution tracker service."""
        logger.info("🚀 Starting Quantum Contribution Tracker...")
        logger.info(f"   Validator: {self.validator_keypair.ss58_address}")
        logger.info(f"   Blockchain: {self.blockchain_url}")
        logger.info(f"   Check Interval: {self.check_interval}s")
        
        # Connect to blockchain
        await self.adapter.connect()
        self.substrate = self.adapter.substrate
        self.running = True
        
        logger.info("✅ Contribution tracker started successfully!")
        
        # Start monitoring loop
        await self.monitor_loop()

    async def stop(self):
        """Stop the contribution tracker service."""
        logger.info("🛑 Stopping Quantum Contribution Tracker...")
        self.running = False
        await self.adapter.disconnect()
        logger.info("✅ Tracker stopped successfully!")

    async def monitor_loop(self):
        """Main monitoring loop - checks for completed quantum jobs."""
        while self.running:
            try:
                await self.check_completed_jobs()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Error in monitor loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def check_completed_jobs(self):
        """Check for newly completed quantum jobs and submit contributions."""
        try:
            # Query all completed jobs from the Quantum pallet
            # In production, this would use indexed events or maintain a local cache
            completed_jobs = await self.get_completed_jobs()
            
            new_jobs = [
                job for job in completed_jobs 
                if job['job_id'] not in self.processed_jobs
            ]
            
            if new_jobs:
                logger.info(f"📋 Found {len(new_jobs)} new completed jobs")
                
                for job in new_jobs:
                    await self.process_job_contribution(job)
            
        except Exception as e:
            logger.error(f"❌ Error checking completed jobs: {e}")

    async def get_completed_jobs(self) -> List[Dict]:
        """
        Query completed quantum jobs from the blockchain.
        Returns list of job dictionaries with metadata.
        """
        completed_jobs = []
        
        try:
            # Query the QuantumJobs storage map
            # This is a simplified version - production would use event filtering
            job_counter = self.substrate.query(
                module='Quantum',
                storage_function='JobCounter',
            )
            
            # Query each job (in production, use pagination)
            for job_id_num in range(max(0, job_counter.value - 100), job_counter.value):
                job_id = f"job_{job_id_num}"
                
                job_data = self.substrate.query(
                    module='Quantum',
                    storage_function='QuantumJobs',
                    params=[job_id.encode()]
                )
                
                if job_data and job_data.value:
                    job_info = job_data.value
                    
                    # Check if job is completed
                    if job_info.get('status') == 'Completed':
                        # Get result for accuracy score
                        result_data = self.substrate.query(
                            module='Quantum',
                            storage_function='QuantumResults',
                            params=[job_id.encode()]
                        )
                        
                        if result_data and result_data.value:
                            completed_jobs.append({
                                'job_id': job_id,
                                'submitter': job_info['submitter'],
                                'executor': job_info.get('executor', job_info['submitter']),
                                'num_qubits': job_info['num_qubits'],
                                'num_shots': job_info['num_shots'],
                                'circuit_depth': self.estimate_circuit_depth(job_info),
                                'accuracy_score': self.calculate_accuracy_score(result_data.value),
                            })
            
        except Exception as e:
            logger.error(f"❌ Error querying completed jobs: {e}")
        
        return completed_jobs

    def estimate_circuit_depth(self, job_info: Dict) -> int:
        """
        Estimate circuit depth from job metadata.
        In production, this would be stored in the job or result data.
        """
        # Simplified estimation based on qubit count
        num_qubits = job_info.get('num_qubits', 5)
        return num_qubits * 10  # Rough estimate

    def calculate_accuracy_score(self, result_data: Dict) -> int:
        """
        Calculate accuracy score from result data.
        Returns score 0-100.
        """
        # Simplified scoring - production would use verification metrics
        if result_data.get('verification_status') == 'Verified':
            return 95
        elif result_data.get('verification_status') == 'Pending':
            return 70
        else:
            return 50

    async def process_job_contribution(self, job: Dict):
        """Process a single job contribution and submit to staking pallet."""
        job_id = job['job_id']
        
        try:
            logger.info(f"⚙️  Processing contribution for job: {job_id}")
            
            # Prepare contribution data
            validator_address = self.validator_keypair.ss58_address
            
            # Submit contribution to staking pallet
            tx_hash = await self.submit_contribution(
                job_id=job_id,
                validator=validator_address,
                num_qubits=job['num_qubits'],
                circuit_depth=job['circuit_depth'],
                num_shots=job['num_shots'],
                accuracy_score=job['accuracy_score'],
            )
            
            # Record contribution
            contribution = ContributionRecord(
                job_id=job_id,
                validator_address=validator_address,
                num_qubits=job['num_qubits'],
                circuit_depth=job['circuit_depth'],
                num_shots=job['num_shots'],
                accuracy_score=job['accuracy_score'],
                submitted_at=datetime.now(),
                tx_hash=tx_hash,
            )
            
            self.contribution_history.append(contribution)
            self.processed_jobs.add(job_id)
            
            # Update statistics
            self.total_contributions += 1
            self.total_qubits_processed += job['num_qubits']
            self.total_shots_executed += job['num_shots']
            
            logger.info(f"✅ Contribution submitted successfully!")
            logger.info(f"   Job ID: {job_id}")
            logger.info(f"   Qubits: {job['num_qubits']}, Depth: {job['circuit_depth']}")
            logger.info(f"   Accuracy: {job['accuracy_score']}%")
            logger.info(f"   TX Hash: {tx_hash[:16]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to process contribution for {job_id}: {e}")

    async def submit_contribution(
        self,
        job_id: str,
        validator: str,
        num_qubits: int,
        circuit_depth: int,
        num_shots: int,
        accuracy_score: int,
    ) -> str:
        """
        Submit quantum contribution to the staking pallet.
        
        Returns:
            Transaction hash
        """
        # Create extrinsic call
        call = self.substrate.compose_call(
            call_module='Staking',
            call_function='record_quantum_contribution',
            call_params={
                'job_id': job_id.encode(),
                'validator': validator,
                'num_qubits': num_qubits,
                'circuit_depth': circuit_depth,
                'num_shots': num_shots,
                'accuracy_score': accuracy_score,
            }
        )
        
        # Create signed extrinsic
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call,
            keypair=self.validator_keypair,
        )
        
        # Submit extrinsic
        receipt = self.substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=True,
        )
        
        if receipt.is_success:
            return receipt.extrinsic_hash
        else:
            raise Exception(f"Extrinsic failed: {receipt.error_message}")

    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        return {
            'total_contributions': self.total_contributions,
            'total_qubits_processed': self.total_qubits_processed,
            'total_shots_executed': self.total_shots_executed,
            'processed_jobs_count': len(self.processed_jobs),
            'contribution_history_size': len(self.contribution_history),
        }

    def get_validator_leaderboard(self) -> List[Dict]:
        """
        Generate leaderboard of validator quantum contributions.
        Queries blockchain for all validator quantum stats.
        """
        leaderboard = []
        
        try:
            # Query all validators
            validators = self.substrate.query_map(
                module='Staking',
                storage_function='Validators',
            )
            
            for validator_id, _ in validators:
                # Get quantum stats for each validator
                stats = self.substrate.query(
                    module='Staking',
                    storage_function='ValidatorQuantumStatsMap',
                    params=[validator_id]
                )
                
                if stats and stats.value:
                    leaderboard.append({
                        'validator': validator_id.value,
                        'jobs_executed': stats.value['jobs_executed'],
                        'avg_accuracy': stats.value['avg_accuracy'],
                        'total_qubits': stats.value['total_qubits'],
                        'quantum_score': stats.value['quantum_score'],
                    })
            
            # Sort by quantum score (descending)
            leaderboard.sort(key=lambda x: x['quantum_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error generating leaderboard: {e}")
        
        return leaderboard


# Example usage
async def main():
    """Example: Run the contribution tracker as a service."""
    
    # Create validator keypair (in production, load from secure keystore)
    validator_keypair = Keypair.create_from_uri('//Alice')
    
    # Create and start tracker
    tracker = QuantumContributionTracker(
        blockchain_url="ws://127.0.0.1:9944",
        validator_keypair=validator_keypair,
        check_interval=10,
    )
    
    try:
        await tracker.start()
    except KeyboardInterrupt:
        logger.info("\n🛑 Received shutdown signal...")
        await tracker.stop()
        
        # Print final statistics
        stats = tracker.get_statistics()
        logger.info("\n📊 Final Statistics:")
        logger.info(f"   Total Contributions: {stats['total_contributions']}")
        logger.info(f"   Total Qubits Processed: {stats['total_qubits_processed']}")
        logger.info(f"   Total Shots Executed: {stats['total_shots_executed']}")


if __name__ == "__main__":
    asyncio.run(main())
