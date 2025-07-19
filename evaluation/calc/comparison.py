import pandas as pd


class MetricComparator:
    """Compare metrics between different agent types for each simulation type"""
    
    def __init__(self):
        self.excluded_metrics = {'agent_type', 'simulation_type', 'total_simulation_timesteps'}
    
    def compare_all_simulations(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare metrics between agent types for all simulation types
        
        Args:
            metrics_df: Multi-indexed DataFrame with metrics indexed by (simulation_type, agent_type)
            
        Returns:
            Long format DataFrame with:
            - Index: simulation_type, metric_name
            - Columns: 'aif_agent', 'heuristic', 'delta'
            - Access pattern: df.loc[(sim_type, metric_name)] returns Series with 3 values
        """
        # Get unique simulation types from the DataFrame
        sim_types = metrics_df.index.get_level_values('simulation_type').unique()
        
        # Lists to store the results directly in long format
        sim_types_list = []
        metric_names_list = []
        aif_values = []
        heuristic_values = []
        delta_values = []
        
        for sim_type in sim_types:
            try:
                # Get metrics for both agents for this simulation type
                aif_metrics = metrics_df.loc[(sim_type, 'active_inference_relative_control')]
                heuristic_metrics = metrics_df.loc[(sim_type, 'heuristic')]
                
                # Process each metric directly
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
                        delta_value = round(percentage_diff, 4)
                    else:
                        # Handle division by zero
                        if aif_value == 0:
                            delta_value = 0.0
                        else:
                            delta_value = float('inf') if aif_value > 0 else float('-inf')
                    
                    # Add to lists
                    sim_types_list.append(sim_type)
                    metric_names_list.append(metric_name)
                    aif_values.append(round(aif_value, 4))
                    heuristic_values.append(round(heuristic_value, 4))
                    delta_values.append(delta_value)
                    
            except KeyError:
                print(f"Warning: Missing data for simulation type '{sim_type}'")
                continue
        
        # Create DataFrame directly in the desired format
        comparison_df = pd.DataFrame({
            'simulation_type': sim_types_list,
            'metric_name': metric_names_list,
            'aif_agent': aif_values,
            'heuristic': heuristic_values,
            'delta': delta_values
        })
        
        # Set multi-index and sort
        comparison_df = comparison_df.set_index(['simulation_type', 'metric_name']).sort_index()

        return comparison_df

def compare_agent_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare agent metrics across all simulation types
    
    Args:
        metrics_df: Multi-indexed DataFrame with simulation metrics indexed by (simulation_type, agent_type)
        
    Returns:
        Long format DataFrame with:
        - Index: simulation_type, metric_name
        - Columns: 'aif_agent', 'heuristic', 'delta'
        - Access pattern: df.loc[('basic', 'time_all_slo_fulfilled_simultaneously_percent')] gives you all 3 values
    """
    comparator = MetricComparator()
    comparison_df = comparator.compare_all_simulations(metrics_df)

    return comparison_df