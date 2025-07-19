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

from evaluation.calc.comparison import compare_agent_metrics
from evaluation.calc.slo_calc import calculate_and_save_slo_metrics
from evaluation.enums.directory_type import DirectoryType
from evaluation.evaluation_utils import EvaluationUtils
from evaluation.plotting.slo_stats_plot import plot_all_slo_stats
from evaluation.plotting.worker_stats_plot import plot_all_worker_stats
from evaluation.simulation.simulation_type import SimulationType
from producer.enums.agent_type import AgentType


def load_consolidated_metrics(filepath: str = 'out/consolidated_metrics.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, index_col=[0, 1])
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


def create_all_metrics_dataframe(metrics_data: list) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics_data)
    # Set multi-index with simulation_type as level 0 and agent_type as level 1
    metrics_df = metrics_df.set_index(['simulation_type', 'agent_type'])
    return metrics_df

def save_comparison_results(comparison_df: pd.DataFrame) -> None:
    """Save the comparison results DataFrame to files"""
    csv_path = EvaluationUtils.get_consolidated_filepath(DirectoryType.OUTPUT, "agent_comparison_results", "csv")
    EvaluationUtils.ensure_directory_exists(csv_path)
    comparison_df.to_csv(csv_path)

def load_simulation_data(agent_type: AgentType, sim_type: SimulationType) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both SLO stats and worker stats for a given agent and simulation type
    
    Args:
        agent_type: The agent type enum
        sim_type: The simulation type enum
        data_dir: Directory where statistics files are stored
        
    Returns:
        Tuple of (slo_stats_df, worker_stats_df) or (None, None) if files not found
    """
    slo_stats_filepath = EvaluationUtils.get_filepath(DirectoryType.SIM_DATA, sim_type, agent_type, "slo_stats", "csv")
    worker_stats_filepath = EvaluationUtils.get_filepath(DirectoryType.SIM_DATA, sim_type, agent_type, "worker_stats", "csv")
    
    try:
        slo_stats_df = pd.read_csv(slo_stats_filepath, index_col=0)
        worker_stats_df = pd.read_csv(worker_stats_filepath, index_col=0)
        return slo_stats_df, worker_stats_df
    except FileNotFoundError as e:
        print(f"Error: Simulation data file not found: {e.filename}")
        return None, None
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        return None, None


def evaluate():
    """Run evaluation pipeline for all simulation data"""
    
    # Configuration using enums
    agent_types = [AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, AgentType.HEURISTIC]
    sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

    metrics_data = []

    # Step 1 & 2: Load simulation data once and create plots, then calculate metrics
    for sim_type in sim_types:
        for agent_type in agent_types:
            # Load simulation data once
            slo_stats_df, worker_stats_df = load_simulation_data(agent_type, sim_type)
            
            if slo_stats_df is None or worker_stats_df is None:
                print(f"Skipping {agent_type.name}/{sim_type.name} due to missing data")
                continue
            
            # Create plots using the loaded data
            plot_all_slo_stats(slo_stats_df, agent_type, sim_type)
            plot_all_worker_stats(worker_stats_df, agent_type, sim_type)
            
            # Calculate metrics using the loaded data
            metrics = calculate_and_save_slo_metrics(slo_stats_df, agent_type.name.lower(), sim_type.name.lower())
            
            metrics_row = metrics.copy()
            metrics_row['simulation_type'] = sim_type.name.lower()
            metrics_row['agent_type'] = agent_type.name.lower()
            metrics_data.append(metrics_row)

    # Step 3: Create multi-indexed DataFrame
    metrics_df = create_all_metrics_dataframe(metrics_data)

    # Step 4: For each simulation type, compare the calculated values between agents
    comparison_df = compare_agent_metrics(metrics_df)
    save_comparison_results(comparison_df)
    

if __name__ == "__main__":
    evaluate()
