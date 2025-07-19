import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType
from ..evaluation_utils import EvaluationUtils
from ..enums.directory_type import DirectoryType


class SloCalculator:
    """Calculator for SLO fulfillment metrics and performance analysis"""
    
    SLO_COLUMNS = [
        'queue_size_slo_value',
        'memory_usage_slo_value', 
        'avg_global_processing_time_slo_value',
        'avg_worker_processing_time_slo_value'
    ]
    
    QUALITY_COLUMNS = [
        'fps_capacity',
        'resolution_capacity',
        'inference_quality_capacity'
    ]
    
    def calculate_all_metrics(self, slo_stats_df: pd.DataFrame, agent_type_name: str, sim_type_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive SLO and performance metrics
        
        Args:
            slo_stats_df: DataFrame containing SLO statistics
            agent_type_name: Name of the agent type
            sim_type_name: Name of the simulation type
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {'agent_type': agent_type_name, 'simulation_type': sim_type_name,
                   'total_simulation_timesteps': len(slo_stats_df),
                   'time_all_slo_fulfilled_simultaneously_percent': self._calculate_all_slo_fullfillment_simultaneously_rate(slo_stats_df),
                   'average_slo_fulfillment_rate_percent': self._calculate_average_slo_fulfillment_rate(slo_stats_df)}
        

        # Individual SLO fulfillment rates
        individual_rates = self._calculate_individual_slo_fulfillment_rates(slo_stats_df)
        metrics.update(individual_rates)
        
        # Additional performance metrics
        additional_metrics = self._calculate_additional_metrics(slo_stats_df)
        metrics.update(additional_metrics)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(slo_stats_df)
        metrics.update(quality_metrics)
        
        # Statistical analysis
        statistical_metrics = self._calculate_slo_statistical_metrics(slo_stats_df)
        metrics.update(statistical_metrics)
        
        return metrics
    
    def _calculate_all_slo_fullfillment_simultaneously_rate(self, slo_stats_df: pd.DataFrame) -> float:
        """
        Calculate the percentage of time when ALL SLOs are fulfilled simultaneously
        """
        # All SLOs are fulfilled when all values <= 1.0
        all_slos_fulfilled = (slo_stats_df[self.SLO_COLUMNS] <= 1.0).all(axis=1)
        fulfillment_rate = all_slos_fulfilled.mean()
        return float(fulfillment_rate)
    
    def _calculate_average_slo_fulfillment_rate(self, slo_stats_df: pd.DataFrame) -> float:
        """
        Calculate fulfillment rate for each SLO individually, then average
        """
        individual_rates = [(slo_stats_df[col] <= 1.0).mean() for col in self.SLO_COLUMNS]
        
        return float(np.mean(individual_rates))
    
    def _calculate_individual_slo_fulfillment_rates(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate fulfillment rate for each individual SLO
        """
        mapping = {
            'queue_size_slo_value': 'queue_size_slo_fulfillment_rate_percent',
            'memory_usage_slo_value': 'memory_usage_slo_fulfillment_rate_percent', 
            'avg_global_processing_time_slo_value': 'global_processing_time_slo_fulfillment_rate_percent',
            'avg_worker_processing_time_slo_value': 'worker_processing_time_slo_fulfillment_rate_percent'
        }
        
        return {
            mapping[col]: float((slo_stats_df[col] <= 1.0).mean())
            for col in self.SLO_COLUMNS
        }
    
    def _calculate_additional_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate additional performance metrics for comparison
        """
        metrics = {}
        
        # SLO violation severity (how much SLOs are exceeded when violated)
        for col in self.SLO_COLUMNS:
            violations = slo_stats_df[col][slo_stats_df[col] > 1.0]
            base_name = col.replace("_slo_value", "")
            if len(violations) > 0:
                metrics[f'{base_name}_avg_violation_severity_factor'] = float(violations.mean() - 1.0)
            else:
                metrics[f'{base_name}_average_violation_severity_factor'] = 0.0

        # System stability metrics using dict comprehension
        stability_metrics = {
            f'{col.replace("_slo_value", "_stability_coefficient_of_variation")}': float(
                (lambda std_dev, mean: std_dev / mean if mean > 0 else 0)(
                    slo_stats_df[col].std(), slo_stats_df[col].mean()
                )
            )
            for col in self.SLO_COLUMNS
        }
        metrics.update(stability_metrics)
        
        # Recovery time analysis (consecutive violations)
        metrics['maximum_consecutive_timesteps_with_slo_violations'] = self._calculate_max_consecutive_violations(slo_stats_df)
        
        return metrics
    
    def _calculate_quality_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate quality metrics (capacity utilization)
        """
        # Overall quality score (average of all quality capacities)
        quality_values = [slo_stats_df[col].mean() for col in self.QUALITY_COLUMNS]
        
        metrics = {'average_overall_stream_quality_score': float(np.mean(quality_values))}
        
        # Calculate average time to stable quality configuration
        avg_time_to_stability = self._calculate_avg_time_to_stable_quality(slo_stats_df)
        metrics['average_timesteps_to_reach_stable_quality_configuration'] = avg_time_to_stability
        
        return metrics
    
    def _calculate_slo_statistical_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical metrics for deeper analysis
        """
        # SLO value statistics using dict comprehension
        metrics = {
            f'average_{col}': float(slo_stats_df[col].mean())
            for col in self.SLO_COLUMNS
        }
        
        # Calculate average SLO value across all 4 SLOs
        all_slo_values = [slo_stats_df[col].mean() for col in self.SLO_COLUMNS]
        metrics['average_slo_value_across_all_slos'] = float(np.mean(all_slo_values))
        
        return metrics
    
    def _calculate_max_consecutive_violations(self, slo_stats_df: pd.DataFrame) -> int:
        """
        Calculate the maximum number of consecutive overall SLO violations
        """
        all_slos_violated = (slo_stats_df[self.SLO_COLUMNS] > 1.0).any(axis=1)
        
        max_consecutive = 0
        current_consecutive = 0
        
        for violated in all_slos_violated:
            if violated:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_avg_time_to_stable_quality(self, slo_stats_df: pd.DataFrame, stability_window: int = 5) -> float:
        """
        Calculate the average number of time steps until a stable quality configuration is reached.
        A stable configuration is when quality capacities haven't changed for the specified window.
        
        Args:
            slo_stats_df: DataFrame containing quality statistics
            stability_window: Number of consecutive unchanged time steps to consider stable (default: 5)
            
        Returns:
            Average time steps to reach stability across all stability periods found
        """
        if len(slo_stats_df) < stability_window:
            return float(len(slo_stats_df))  # Not enough data for stability analysis
        
        # Track when quality values change by comparing consecutive rows
        quality_df = slo_stats_df[self.QUALITY_COLUMNS]
        
        # Find points where any quality metric changes
        changes = (quality_df.diff().abs() > 1e-6).any(axis=1)  # Use small threshold for float comparison
        
        stability_periods = []
        current_stable_start = None
        consecutive_stable_count = 0
        
        for i, has_change in enumerate(changes):
            if i == 0:  # Skip first row (diff is always NaN)
                consecutive_stable_count = 1
                current_stable_start = 0
                continue
                
            if not has_change:  # No change detected
                consecutive_stable_count += 1
                
                # Check if we've reached stability
                if consecutive_stable_count >= stability_window and current_stable_start is not None:
                    # We've found a stable period
                    time_to_stability = i - stability_window + 1 - current_stable_start
                    if time_to_stability >= 0:
                        stability_periods.append(time_to_stability)
                    # Reset for next period
                    current_stable_start = None
                    consecutive_stable_count = 0
            else:  # Change detected
                consecutive_stable_count = 0
                current_stable_start = i
        
        # If we ended in a stable state, count it
        if consecutive_stable_count >= stability_window and current_stable_start is not None:
            time_to_stability = len(slo_stats_df) - stability_window - current_stable_start
            if time_to_stability >= 0:
                stability_periods.append(time_to_stability)
        
        # Return average time to stability, or total time if no stability found
        if stability_periods:
            return float(np.mean(stability_periods))
        else:
            return float(len(slo_stats_df))  # No stable periods found
    
    def save_metrics(self, metrics: Dict[str, Any], output_dir: str = "out/metrics"):
        """
        Save calculated metrics to JSON file
        
        Args:
            metrics: Dictionary containing all calculated metrics
            output_dir: Directory to save metrics files
        """
        agent_type = metrics['agent_type']
        sim_type = metrics['simulation_type']
        
        # Convert string names to enum instances if needed
        if isinstance(agent_type, str):
            agent_type = AgentType[agent_type.upper()]
        if isinstance(sim_type, str):
            sim_type = SimulationType[sim_type.upper()]
        
        filepath = EvaluationUtils.get_filepath(DirectoryType.METRICS, sim_type, agent_type, "metrics", "json")
        
        # Ensure directory exists
        EvaluationUtils.ensure_directory_exists(filepath)
        
        # Format metrics for better readability
        formatted_metrics = self._format_metrics_for_output(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(formatted_metrics, f, indent=2)

    def _format_metrics_for_output(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics for better readability in output
        """
        # Round float values to 4 decimal places for readability using dict comprehension
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in metrics.items()
        }


def calculate_and_save_slo_metrics(slo_stats_df: pd.DataFrame, agent_type_name: str, sim_type_name: str) -> Dict[str, Any]:
    """
    Convenience function to calculate and save SLO metrics
    
    Args:
        slo_stats_df: DataFrame containing SLO statistics
        agent_type_name: Name of the agent type
        sim_type_name: Name of the simulation type
        
    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = SloCalculator()
    metrics = calculator.calculate_all_metrics(slo_stats_df, agent_type_name, sim_type_name)
    calculator.save_metrics(metrics)

    return metrics