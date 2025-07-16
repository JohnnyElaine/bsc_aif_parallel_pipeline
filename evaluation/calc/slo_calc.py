import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any


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
                   'total_timesteps': len(slo_stats_df),
                   'overall_slo_fulfillment_rate': self._calculate_overall_slo_fulfillment_rate(slo_stats_df),
                   'average_slo_fulfillment_rate': self._calculate_average_slo_fulfillment_rate(slo_stats_df)}
        

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
    
    def _calculate_overall_slo_fulfillment_rate(self, slo_stats_df: pd.DataFrame) -> float:
        """
        Calculate the percentage of time when ALL SLOs are fulfilled simultaneously
        """
        # All SLOs are fulfilled when all values <= 1.0
        all_slos_fulfilled = (slo_stats_df[self.SLO_COLUMNS] <= 1.0).all(axis=1)
        fulfillment_rate = all_slos_fulfilled.mean()
        return float(fulfillment_rate)
    
    def _calculate_average_slo_fulfillment_rate(self, slo_stats_df: pd.DataFrame) -> float:
        """
        Calculate the average SLO fulfillment rate across all SLOs
        """
        # Calculate fulfillment rate for each SLO individually, then average
        individual_rates = [(slo_stats_df[col] <= 1.0).mean() for col in self.SLO_COLUMNS]
        
        return float(np.mean(individual_rates))
    
    def _calculate_individual_slo_fulfillment_rates(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate fulfillment rate for each individual SLO
        """
        return {
            col.replace('_slo_value', '_fulfillment_rate'): float((slo_stats_df[col] <= 1.0).mean())
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
                metrics[f'{base_name}_avg_violation_severity'] = float(violations.mean() - 1.0)
                metrics[f'{base_name}_max_violation_severity'] = float(violations.max() - 1.0)
            else:
                metrics[f'{base_name}_avg_violation_severity'] = 0.0
                metrics[f'{base_name}_max_violation_severity'] = 0.0
        
        # System stability metrics using dict comprehension
        stability_metrics = {
            f'{col.replace("_slo_value", "_stability_coefficient")}': float(
                (lambda std_dev, mean: std_dev / mean if mean > 0 else 0)(
                    slo_stats_df[col].std(), slo_stats_df[col].mean()
                )
            )
            for col in self.SLO_COLUMNS
        }
        metrics.update(stability_metrics)
        
        # Recovery time analysis (consecutive violations)
        metrics['max_consecutive_overall_violations'] = self._calculate_max_consecutive_violations(slo_stats_df)
        
        return metrics
    
    def _calculate_quality_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate quality metrics (capacity utilization)
        """
        # Overall quality score (average of all quality capacities)
        quality_values = [slo_stats_df[col].mean() for col in self.QUALITY_COLUMNS]
        
        return {'avg_stream_quality': float(np.mean(quality_values))}
    
    def _calculate_slo_statistical_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical metrics for deeper analysis
        """
        # SLO value statistics using dict comprehension
        return {
            f'{col}_mean': float(slo_stats_df[col].mean())
            for col in self.SLO_COLUMNS
        }
    
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
    
    def save_metrics(self, metrics: Dict[str, Any], output_dir: str = "out/metrics"):
        """
        Save calculated metrics to JSON file
        
        Args:
            metrics: Dictionary containing all calculated metrics
            output_dir: Directory to save metrics files
        """
        agent_type = metrics['agent_type']
        sim_type = metrics['simulation_type']
        
        dir_path = os.path.join(output_dir, f'{sim_type}_sim')
        os.makedirs(dir_path, exist_ok=True)
        
        filename = f'{agent_type}_metrics.json'
        filepath = os.path.join(dir_path, filename)
        
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
