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
            Long format DataFrame with:
            - Index: simulation_type, value_type
            - Columns: Each metric as a separate column
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
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.set_index('simulation_type')

        # Convert from wide to long format
        print(comparison_df)
        comparison_df_long = self._convert_to_long_format(comparison_df)
        print(comparison_df_long)
        return comparison_df_long
    
    def _convert_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the wide format DataFrame to long format for better analysis
        
        Args:
            df: Wide format DataFrame with columns like 'metric_name_agent_type'
            
        Returns:
            Long format DataFrame with separate columns for metric, agent_type, and value_type
        """
        # Extract unique metric names by removing the suffixes
        metrics = set()
        for col in df.columns:
            # Remove the suffixes '_aif_agent', '_heuristic', '_delta'
            if col.endswith('_aif_agent'):
                metrics.add(col.replace('_aif_agent', ''))
            elif col.endswith('_heuristic'):
                metrics.add(col.replace('_heuristic', ''))
            elif col.endswith('_delta'):
                metrics.add(col.replace('_delta', ''))
        
        # Create lists to store the reshaped data
        sim_types = []
        value_types = []
        metric_data = {metric: [] for metric in metrics}
        
        # Iterate through each simulation type (index)
        for sim_type in df.index:
            # For each value type (aif_agent, heuristic, delta)
            for value_type in ['aif_agent', 'heuristic', 'delta']:
                sim_types.append(sim_type)
                value_types.append(value_type)
                
                # For each metric, get the corresponding value
                for metric in metrics:
                    col_name = f'{metric}_{value_type}'
                    if col_name in df.columns:
                        metric_data[metric].append(df.loc[sim_type, col_name])
                    else:
                        metric_data[metric].append(None)
        
        # Create the long format DataFrame
        long_data = {
            'simulation_type': sim_types,
            'value_type': value_types,
        }
        # Add all the metric columns
        long_data.update(metric_data)
        
        # Create DataFrame and set multi-index
        df_long = pd.DataFrame(long_data)
        df_final = df_long.set_index(['simulation_type', 'value_type']).sort_index()
        
        return df_final
    
    def calculate_percentage_differences(self, aif_metrics: pd.Series, heuristic_metrics: pd.Series, sim_type: str) -> Dict[str, Any]:
        """
        Calculate percentage difference between AIF and Heuristic agent metrics
        Also includes the original values from both agents
        
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
            
            # Store original values
            differences[f'{metric_name}_aif_agent'] = round(aif_value, 4)
            differences[f'{metric_name}_heuristic'] = round(heuristic_value, 4)
            
            # Calculate percentage difference
            if heuristic_value != 0:
                percentage_diff = ((aif_value - heuristic_value) / heuristic_value) * 100
                differences[f'{metric_name}_delta'] = round(percentage_diff, 4)
            else:
                # Handle division by zero
                if aif_value == 0:
                    differences[f'{metric_name}_delta'] = 0.0
                else:
                    differences[f'{metric_name}_delta'] = float('inf') if aif_value > 0 else float('-inf')
        
        return differences


def compare_agent_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare agent metrics across all simulation types
    
    Args:
        metrics_df: Multi-indexed DataFrame with simulation metrics
        
    Returns:
        Long format DataFrame with:
        - Index: simulation_type, value_type (aif_agent/heuristic/delta)
        - Columns: Each metric as a separate column
    """
    comparator = MetricComparator()
    comparison_df = comparator.compare_all_simulations(metrics_df)

    return comparison_df