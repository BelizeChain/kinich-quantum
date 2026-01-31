"""
Result Aggregator for Kinich

Aggregates and analyzes results from multiple quantum job executions,
providing statistical analysis, error mitigation, and confidence scoring.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import Counter
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Aggregated quantum job result."""
    
    # Primary results
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Statistical measures
    mean_fidelity: float = 0.0
    std_fidelity: float = 0.0
    confidence_score: float = 0.0
    
    # Execution details
    total_shots: int = 0
    num_executions: int = 0
    backends_used: List[str] = field(default_factory=list)
    
    # Error mitigation
    error_mitigated: bool = False
    raw_counts: Optional[Dict[str, int]] = None
    
    # Metadata
    aggregated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ResultAggregator:
    """
    Aggregates results from multiple quantum executions.
    
    Features:
    - Multi-execution result combining
    - Statistical analysis
    - Error mitigation
    - Confidence scoring
    - Outlier detection
    - Result validation
    """
    
    def __init__(self):
        """Initialize result aggregator."""
        self._total_aggregations = 0
        self._results_processed = 0
        
        logger.info("Initialized result aggregator")
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        apply_error_mitigation: bool = True
    ) -> AggregatedResult:
        """
        Aggregate multiple quantum execution results.
        
        Args:
            results: List of execution results
            apply_error_mitigation: Apply error mitigation
        
        Returns:
            Aggregated result
        """
        if not results:
            logger.warning("No results to aggregate")
            return AggregatedResult()
        
        logger.info(f"Aggregating {len(results)} execution results")
        
        # Combine counts from all executions
        combined_counts: Dict[str, int] = {}
        total_shots = 0
        backends_used = []
        fidelities = []
        
        for result in results:
            # Get counts
            counts = result.get('counts', {})
            
            for state, count in counts.items():
                combined_counts[state] = combined_counts.get(state, 0) + count
                total_shots += count
            
            # Track backends
            backend = result.get('backend_used', result.get('backend', 'unknown'))
            if backend not in backends_used:
                backends_used.append(backend)
            
            # Track fidelity
            if 'fidelity' in result:
                fidelities.append(result['fidelity'])
        
        # Apply error mitigation if requested
        if apply_error_mitigation:
            mitigated_counts = self._apply_error_mitigation(combined_counts, total_shots)
            raw_counts = combined_counts.copy()
            combined_counts = mitigated_counts
            error_mitigated = True
        else:
            raw_counts = None
            error_mitigated = False
        
        # Calculate probabilities
        probabilities = {
            state: count / total_shots
            for state, count in combined_counts.items()
        } if total_shots > 0 else {}
        
        # Calculate statistical measures
        mean_fidelity = statistics.mean(fidelities) if fidelities else 0.0
        std_fidelity = statistics.stdev(fidelities) if len(fidelities) > 1 else 0.0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            results, 
            combined_counts, 
            total_shots
        )
        
        # Update statistics
        self._total_aggregations += 1
        self._results_processed += len(results)
        
        return AggregatedResult(
            counts=combined_counts,
            probabilities=probabilities,
            mean_fidelity=mean_fidelity,
            std_fidelity=std_fidelity,
            confidence_score=confidence,
            total_shots=total_shots,
            num_executions=len(results),
            backends_used=backends_used,
            error_mitigated=error_mitigated,
            raw_counts=raw_counts
        )
    
    def _apply_error_mitigation(
        self,
        counts: Dict[str, int],
        total_shots: int
    ) -> Dict[str, int]:
        """
        Apply simple error mitigation to counts.
        
        Uses measurement error mitigation by filtering low-probability states.
        """
        if not counts:
            return counts
        
        # Calculate minimum count threshold (0.5% of total shots)
        min_threshold = max(1, int(total_shots * 0.005))
        
        # Filter out states below threshold (likely measurement errors)
        mitigated = {
            state: count
            for state, count in counts.items()
            if count >= min_threshold
        }
        
        # If we filtered everything, keep original
        if not mitigated:
            return counts
        
        # Renormalize to preserve total shots
        current_total = sum(mitigated.values())
        if current_total != total_shots:
            scale_factor = total_shots / current_total
            mitigated = {
                state: int(count * scale_factor)
                for state, count in mitigated.items()
            }
        
        logger.debug(
            f"Error mitigation: {len(counts)} states → {len(mitigated)} states"
        )
        
        return mitigated
    
    def _calculate_confidence(
        self,
        results: List[Dict[str, Any]],
        combined_counts: Dict[str, int],
        total_shots: int
    ) -> float:
        """
        Calculate confidence score for aggregated results.
        
        Based on:
        - Number of executions
        - Result consistency across executions
        - Total shots
        - Backend diversity
        """
        if not results:
            return 0.0
        
        confidence = 0.0
        
        # 1. Execution count factor (0-25 points)
        num_executions = len(results)
        execution_score = min(25, num_executions * 5)
        confidence += execution_score
        
        # 2. Shot count factor (0-25 points)
        shots_score = min(25, total_shots / 100)  # 100 shots = 1 point
        confidence += shots_score
        
        # 3. Consistency factor (0-30 points)
        consistency = self._calculate_consistency(results)
        confidence += consistency * 30
        
        # 4. Backend diversity factor (0-20 points)
        unique_backends = len(set(
            r.get('backend_used', r.get('backend', 'unknown'))
            for r in results
        ))
        diversity_score = min(20, unique_backends * 10)
        confidence += diversity_score
        
        # Normalize to 0-1 range
        return min(1.0, confidence / 100)
    
    def _calculate_consistency(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency across multiple results.
        
        Returns value between 0 (inconsistent) and 1 (very consistent).
        """
        if len(results) < 2:
            return 1.0
        
        # Get top state from each result
        top_states = []
        for result in results:
            counts = result.get('counts', {})
            if counts:
                top_state = max(counts.items(), key=lambda x: x[1])[0]
                top_states.append(top_state)
        
        if not top_states:
            return 0.0
        
        # Calculate how often the most common state appears
        state_counter = Counter(top_states)
        most_common_count = state_counter.most_common(1)[0][1]
        
        consistency = most_common_count / len(top_states)
        return consistency
    
    def compare_results(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any]
    ) -> float:
        """
        Compare similarity between two results.
        
        Args:
            result1: First result
            result2: Second result
        
        Returns:
            Similarity score (0-1)
        """
        counts1 = result1.get('counts', {})
        counts2 = result2.get('counts', {})
        
        if not counts1 or not counts2:
            return 0.0
        
        # Get all unique states
        all_states = set(counts1.keys()) | set(counts2.keys())
        
        # Calculate total shots for each
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # Calculate probability distributions
        prob1 = {state: counts1.get(state, 0) / total1 for state in all_states}
        prob2 = {state: counts2.get(state, 0) / total2 for state in all_states}
        
        # Calculate fidelity (using classical fidelity)
        fidelity = sum(
            (prob1[state] * prob2[state]) ** 0.5
            for state in all_states
        ) ** 2
        
        return fidelity
    
    def detect_outliers(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> List[int]:
        """
        Detect outlier results that differ significantly from others.
        
        Args:
            results: List of results
            threshold: Similarity threshold (below this = outlier)
        
        Returns:
            Indices of outlier results
        """
        if len(results) < 3:
            return []
        
        outliers = []
        
        for i, result in enumerate(results):
            # Compare with all other results
            similarities = []
            
            for j, other_result in enumerate(results):
                if i != j:
                    similarity = self.compare_results(result, other_result)
                    similarities.append(similarity)
            
            # If average similarity is below threshold, mark as outlier
            avg_similarity = statistics.mean(similarities) if similarities else 0.0
            
            if avg_similarity < threshold:
                outliers.append(i)
                logger.warning(
                    f"Result {i} detected as outlier "
                    f"(similarity: {avg_similarity:.3f})"
                )
        
        return outliers
    
    def filter_outliers(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Remove outlier results from list.
        
        Args:
            results: List of results
            threshold: Similarity threshold
        
        Returns:
            Filtered results
        """
        outlier_indices = self.detect_outliers(results, threshold)
        
        if not outlier_indices:
            return results
        
        filtered = [
            result for i, result in enumerate(results)
            if i not in outlier_indices
        ]
        
        logger.info(
            f"Filtered {len(outlier_indices)} outliers, "
            f"{len(filtered)} results remain"
        )
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        avg_results_per_aggregation = (
            self._results_processed / self._total_aggregations
            if self._total_aggregations > 0 else 0
        )
        
        return {
            'total_aggregations': self._total_aggregations,
            'results_processed': self._results_processed,
            'avg_results_per_aggregation': avg_results_per_aggregation,
        }
