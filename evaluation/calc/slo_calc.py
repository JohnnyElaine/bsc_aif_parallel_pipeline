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
        metrics = {
            'agent_type': agent_type_name,
            'simulation_type': sim_type_name,
            'total_timesteps': len(slo_stats_df)
        }
        
        # 1. Overall SLO fulfillment rate (all 4 SLOs fulfilled simultaneously)
        metrics['overall_slo_fulfillment_rate'] = self._calculate_overall_slo_fulfillment_rate(slo_stats_df)
        
        # 2. Average SLO fulfillment rate
        metrics['average_slo_fulfillment_rate'] = self._calculate_average_slo_fulfillment_rate(slo_stats_df)
        
        # 3. Individual SLO fulfillment rates
        individual_rates = self._calculate_individual_slo_fulfillment_rates(slo_stats_df)
        metrics.update(individual_rates)
        
        # 4. Additional performance metrics
        additional_metrics = self._calculate_additional_metrics(slo_stats_df)
        metrics.update(additional_metrics)
        
        # 5. Quality metrics
        quality_metrics = self._calculate_quality_metrics(slo_stats_df)
        metrics.update(quality_metrics)
        
        # 6. Statistical analysis
        statistical_metrics = self._calculate_statistical_metrics(slo_stats_df)
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
        individual_rates = []
        for col in self.SLO_COLUMNS:
            rate = (slo_stats_df[col] <= 1.0).mean()
            individual_rates.append(rate)
        
        return float(np.mean(individual_rates))
    
    def _calculate_individual_slo_fulfillment_rates(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate fulfillment rate for each individual SLO
        """
        rates = {}
        for col in self.SLO_COLUMNS:
            rate = (slo_stats_df[col] <= 1.0).mean()
            metric_name = col.replace('_slo_value', '_fulfillment_rate')
            rates[metric_name] = float(rate)
        return rates
    
    def _calculate_additional_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate additional performance metrics for comparison
        """
        metrics = {}
        
        # SLO violation severity (how much SLOs are exceeded when violated)
        for col in self.SLO_COLUMNS:
            violations = slo_stats_df[col][slo_stats_df[col] > 1.0]
            if len(violations) > 0:
                avg_violation_severity = violations.mean() - 1.0  # How much over 1.0
                max_violation_severity = violations.max() - 1.0
                metrics[f'{col.replace("_slo_value", "_avg_violation_severity")}'] = float(avg_violation_severity)
                metrics[f'{col.replace("_slo_value", "_max_violation_severity")}'] = float(max_violation_severity)
            else:
                metrics[f'{col.replace("_slo_value", "_avg_violation_severity")}'] = 0.0
                metrics[f'{col.replace("_slo_value", "_max_violation_severity")}'] = 0.0
        
        # System stability metrics
        for col in self.SLO_COLUMNS:
            std_dev = slo_stats_df[col].std()
            coefficient_of_variation = std_dev / slo_stats_df[col].mean() if slo_stats_df[col].mean() > 0 else 0
            metrics[f'{col.replace("_slo_value", "_stability_coefficient")}'] = float(coefficient_of_variation)
        
        # Recovery time analysis (consecutive violations)
        metrics['max_consecutive_overall_violations'] = self._calculate_max_consecutive_violations(slo_stats_df)
        
        # Resource utilization efficiency
        metrics['avg_queue_size'] = float(slo_stats_df['queue_size'].mean())
        metrics['max_queue_size'] = float(slo_stats_df['queue_size'].max())
        metrics['avg_memory_usage'] = float(slo_stats_df['memory_usage'].mean())
        metrics['max_memory_usage'] = float(slo_stats_df['memory_usage'].max())
        
        return metrics
    
    def _calculate_quality_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate quality metrics (capacity utilization)
        """
        metrics = {}
        for col in self.QUALITY_COLUMNS:
            avg_capacity = slo_stats_df[col].mean()
            min_capacity = slo_stats_df[col].min()
            max_capacity = slo_stats_df[col].max()
            std_capacity = slo_stats_df[col].std()
            
            metrics[f'avg_{col}'] = float(avg_capacity)
            metrics[f'min_{col}'] = float(min_capacity)
            metrics[f'max_{col}'] = float(max_capacity)
            metrics[f'std_{col}'] = float(std_capacity)
        
        # Overall quality score (average of all quality capacities)
        quality_values = []
        for col in self.QUALITY_COLUMNS:
            quality_values.append(slo_stats_df[col].mean())
        
        metrics['overall_quality_score'] = float(np.mean(quality_values))
        
        return metrics
    
    def _calculate_statistical_metrics(self, slo_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical metrics for deeper analysis
        """
        metrics = {}
        
        # SLO value statistics
        for col in self.SLO_COLUMNS:
            metrics[f'{col}_mean'] = float(slo_stats_df[col].mean())
            metrics[f'{col}_median'] = float(slo_stats_df[col].median())
            metrics[f'{col}_std'] = float(slo_stats_df[col].std())
            metrics[f'{col}_95th_percentile'] = float(slo_stats_df[col].quantile(0.95))
            metrics[f'{col}_99th_percentile'] = float(slo_stats_df[col].quantile(0.99))
        
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
    
    def save_metrics(self, metrics: Dict[str, Any], output_dir: str = "out/metrics"):
        """
        Save calculated metrics to JSON file
        
        Args:
            metrics: Dictionary containing all calculated metrics
            output_dir: Directory to save metrics files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        agent_type = metrics['agent_type']
        sim_type = metrics['simulation_type']
        filename = f"{sim_type}_{agent_type}_metrics.json"
        filepath = os.path.join(output_dir, filename)
        
        # Format metrics for better readability
        formatted_metrics = self._format_metrics_for_output(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(formatted_metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    
    def _format_metrics_for_output(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics for better readability in output
        """
        formatted = {}
        
        # Round float values to 4 decimal places for readability
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted[key] = round(value, 4)
            else:
                formatted[key] = value
        
        return formatted


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
