"""
Evaluation Script for Simulation Data Analysis

This script handles all the analysis tasks after simulation data has been saved:
1. Read the simulation data
2. Create plots for each simulation and agent using slo_stats_plot
3. Calculate values using slo_calc.py for each simulation and agent
4. For each simulation type: compare the calculated values for active inference and heuristic agent using comparison.py

Usage:
    python evaluate_sim_data.py
"""

import pandas as pd
from evaluation.plotting.slo_stats_plot import plot_all_slo_stats_from_file
from evaluation.plotting.worker_stats_plot import plot_all_worker_stats_from_file
from evaluation.calc.slo_calc import calculate_and_save_slo_metrics_from_file
from evaluation.calc.comparison import compare_agent_metrics
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType


def load_consolidated_metrics(filepath: str = 'out/consolidated_metrics.parquet') -> pd.DataFrame:
    try:
        df = pd.read_parquet(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: Consolidated metrics file not found: {filepath}")
        print("Run evaluate() first to generate the consolidated metrics.")
        return None
    except Exception as e:
        print(f"Error loading consolidated metrics: {e}")
        return None


def get_metrics_for_combination(sim_type: SimulationType, agent_type: AgentType, 
                               metrics_df: pd.DataFrame = None) -> pd.Series:
    if metrics_df is None:
        metrics_df = load_consolidated_metrics()
        if metrics_df is None:
            return None
    
    sim_name = sim_type.name.lower()
    agent_name = agent_type.name.lower()
    
    try:
        return metrics_df.loc[(sim_name, agent_name)]
    except KeyError:
        print(f"Error: No metrics found for {sim_name}/{agent_name} combination")
        return None


def create_consolidated_metrics_dataframe(metrics_data: list) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics_data)
    # Set multi-index with simulation_type as level 0 and agent_type as level 1
    metrics_df = metrics_df.set_index(['simulation_type', 'agent_type'])
    return metrics_df

def save_consolidated_metrics(metrics_df: pd.DataFrame, output_dir: str = 'out') -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_path = os.path.join(output_dir, 'consolidated_metrics.parquet')
    csv_path = os.path.join(output_dir, 'consolidated_metrics.csv')
    
    metrics_df.to_parquet(parquet_path)
    metrics_df.to_csv(csv_path)


def evaluate():
    """Run evaluation pipeline for all simulation data"""
    
    # Configuration using enums
    agent_types = [AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, AgentType.HEURISTIC]
    sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]
    
    # Step 1 & 2: Read simulation data and create plots for each simulation and agent
    for sim_type in sim_types:
        for agent_type in agent_types:
            plot_all_slo_stats_from_file(agent_type, sim_type)
            plot_all_worker_stats_from_file(agent_type, sim_type)
    
    # Step 3: Calculate values using slo_calc.py for each simulation and agent
    # Create a multi-indexed DataFrame to store all metrics
    metrics_data = []
    
    for sim_type in sim_types:
        for agent_type in agent_types:
            # Calculate and get the metrics
            metrics = calculate_and_save_slo_metrics_from_file(agent_type, sim_type)
            
            if metrics is not None:
                # Create a row for the DataFrame with simulation type and agent type as index
                metrics_row = metrics.copy()
                metrics_row['simulation_type'] = sim_type.name.lower()
                metrics_row['agent_type'] = agent_type.name.lower()
                metrics_data.append(metrics_row)
    
    # Create multi-indexed DataFrame
    metrics_df = create_consolidated_metrics_dataframe(metrics_data)
    save_consolidated_metrics(metrics_df)

    # Step 4: For each simulation type, compare the calculated values between agents
    compare_agent_metrics()
    
    return metrics_df


if __name__ == "__main__":
    consolidated_metrics_df = evaluate()
