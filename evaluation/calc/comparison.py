import pandas as pd
from typing import Dict, Any
from evaluation.simulation.simulation_type import SimulationType
from producer.enums.agent_type import AgentType


class MetricComparator:
    """Compare metrics between different agent types for each simulation type"""
    
    def __init__(self):
        self.excluded_metrics = {'agent_type', 'simulation_type', 'total_simulation_timesteps'}
    
    def compare_all_simulations(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare metrics between agent types for all simulation types
        
        Args:
            metrics_df: Multi-indexed DataFrame with metrics
            
        Returns:
            DataFrame with percentage differences between AIF and Heuristic agents
        """
        comparison_results = []
        
        # Get unique simulation types from the DataFrame
        sim_types = metrics_df.index.get_level_values('simulation_type').unique()
        
        for sim_type in sim_types:
            try:
                # Get metrics for both agents for this simulation type
                aif_metrics = metrics_df.loc[(sim_type, 'active_inference_relative_control')]
                heuristic_metrics = metrics_df.loc[(sim_type, 'heuristic')]
                
                # Calculate percentage differences
                differences = self.calculate_percentage_differences(aif_metrics, heuristic_metrics, sim_type)
                comparison_results.append(differences)
                
            except KeyError:
                print(f"Warning: Missing data for simulation type '{sim_type}'")
                continue
        
        # Create DataFrame with results
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.set_index('simulation_type')
            return comparison_df
        else:
            return pd.DataFrame()
    
    def calculate_percentage_differences(self, aif_metrics: pd.Series, heuristic_metrics: pd.Series, sim_type: str) -> Dict[str, Any]:
        """
        Calculate percentage difference between AIF and Heuristic agent metrics
        
        Formula: ((AIF_value - Heuristic_value) / Heuristic_value) * 100
        """
        differences = {'simulation_type': sim_type}
        
        # Get all metrics except excluded ones
        for metric_name in aif_metrics.index:
            if metric_name in self.excluded_metrics:
                continue
                
            aif_value = aif_metrics[metric_name]
            heuristic_value = heuristic_metrics[metric_name]
            
            # Skip if either value is missing or not numeric
            if (pd.isna(aif_value) or pd.isna(heuristic_value) or 
                not isinstance(aif_value, (int, float)) or 
                not isinstance(heuristic_value, (int, float))):
                continue
            
            # Calculate percentage difference
            if heuristic_value != 0:
                percentage_diff = ((aif_value - heuristic_value) / heuristic_value) * 100
                differences[f'{metric_name}_percent_diff'] = round(percentage_diff, 4)
            else:
                # Handle division by zero
                if aif_value == 0:
                    differences[f'{metric_name}_percent_diff'] = 0.0
                else:
                    differences[f'{metric_name}_percent_diff'] = float('inf') if aif_value > 0 else float('-inf')
        
        return differences


def compare_agent_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare agent metrics across all simulation types
    
    Args:
        metrics_df: Multi-indexed DataFrame with simulation metrics
        
    Returns:
        DataFrame with percentage differences between agents
    """
    if metrics_df is None or metrics_df.empty:
        print("Error: No metrics data provided for comparison")
        return pd.DataFrame()
    
    comparator = MetricComparator()
    comparison_df = comparator.compare_all_simulations(metrics_df)
    
    if not comparison_df.empty:
        print(f"Comparison completed for {len(comparison_df)} simulation types")
        return comparison_df
    else:
        print("Warning: No comparison results generated")
        return pd.DataFrame()


if __name__ == "__main__":
    compare_agent_metrics()
