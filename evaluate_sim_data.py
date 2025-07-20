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
import seaborn as sns

from evaluation.calc.comparison import compare_agent_metrics
from evaluation.calc.slo_calc import calculate_and_save_slo_metrics
from evaluation.enums.directory_type import DirectoryType
from evaluation.evaluation_utils import EvaluationUtils
from evaluation.latex.comparison_latex import generate_and_save_latex_tables
from evaluation.plotting.slo_stats_plot import plot_all_slo_stats, create_all_combined_plots
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


def create_consolidated_dataframe(data_dict: dict, data_type: str = "slo_stats") -> pd.DataFrame:
    """
    Create a consolidated multi-indexed DataFrame from simulation data
    
    Args:
        data_dict: Dictionary with keys as (sim_type, agent_type) tuples and DataFrames as values
        data_type: Type of data for naming purposes ("slo_stats" or "worker_stats")
    
    Returns:
        Multi-indexed DataFrame with sim_type and agent_type as index levels
    """
    consolidated_data = []
    
    for (sim_type, agent_type), df in data_dict.items():
        # Add multi-index columns to identify the source
        df_copy = df.copy()
        df_copy['simulation_type'] = sim_type
        df_copy['agent_type'] = agent_type
        consolidated_data.append(df_copy)
    
    # Concatenate all DataFrames
    if consolidated_data:
        result_df = pd.concat(consolidated_data, ignore_index=True)
        # Set multi-index
        result_df = result_df.set_index(['simulation_type', 'agent_type'])
        return result_df
    else:
        return pd.DataFrame()

def save_comparison_results(comparison_df: pd.DataFrame) -> None:
    """Save the comparison results DataFrame to files (currently disabled as not used elsewhere)"""
    # CSV saving disabled as the file is not used elsewhere in the codebase
    # csv_path = EvaluationUtils.get_consolidated_filepath(DirectoryType.OUTPUT, "agent_comparison_results", "csv")
    # EvaluationUtils.ensure_directory_exists(csv_path)
    # comparison_df.to_csv(csv_path)
    pass


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


def load_all_simulation_data() -> tuple[dict, dict]:
    """
    Load all simulation data for all combinations of simulation types and agent types
    
    Returns:
        Tuple of (slo_stats_data, worker_stats_data) dictionaries with keys as (sim_type, agent_type)
    """
    # Configuration using enums
    agent_types = [AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, AgentType.HEURISTIC]
    sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

    slo_stats_data = {}
    worker_stats_data = {}
    
    for sim_type in sim_types:
        for agent_type in agent_types:
            # Load simulation data
            slo_stats_df, worker_stats_df = load_simulation_data(agent_type, sim_type)
            
            if slo_stats_df is None or worker_stats_df is None:
                print(f"Skipping {agent_type.name}/{sim_type.name} due to missing data")
                continue
            
            # Store data with multi-index keys
            key = (sim_type.name.lower(), agent_type.name.lower())
            slo_stats_data[key] = slo_stats_df
            worker_stats_data[key] = worker_stats_df
    
    return slo_stats_data, worker_stats_data


def create_simulation_data_dictionaries(slo_stats_data: dict, worker_stats_data: dict) -> tuple[dict, dict]:
    """
    Create consolidated dictionaries from raw simulation data for easy access
    
    Args:
        slo_stats_data: Dictionary of SLO stats DataFrames
        worker_stats_data: Dictionary of worker stats DataFrames
        
    Returns:
        Tuple of (consolidated_slo_stats, consolidated_worker_stats) dictionaries
    """
    consolidated_slo_stats = {}
    consolidated_worker_stats = {}
    
    for (sim_type_name, agent_type_name), slo_df in slo_stats_data.items():
        consolidated_slo_stats[(sim_type_name, agent_type_name)] = slo_df
    
    for (sim_type_name, agent_type_name), worker_df in worker_stats_data.items():
        consolidated_worker_stats[(sim_type_name, agent_type_name)] = worker_df
    
    return consolidated_slo_stats, consolidated_worker_stats


def create_and_save_simulation_dataframes(consolidated_slo_stats: dict, consolidated_worker_stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create multi-indexed DataFrames from simulation data (no longer saves CSV files as they're not used)
    
    Args:
        consolidated_slo_stats: Dictionary of SLO stats DataFrames
        consolidated_worker_stats: Dictionary of worker stats DataFrames
        
    Returns:
        Tuple of (big_slo_stats_df, big_worker_stats_df)
    """
    # Create big consolidated DataFrames
    big_slo_stats_df = create_consolidated_dataframe(consolidated_slo_stats, "slo_stats")
    big_worker_stats_df = create_consolidated_dataframe(consolidated_worker_stats, "worker_stats")
    
    print("Consolidated DataFrames created (not saving CSV files as they're not used)")
    
    return big_slo_stats_df, big_worker_stats_df


def generate_plots_and_calculate_metrics(consolidated_slo_stats: dict, consolidated_worker_stats: dict) -> list:
    """
    Generate plots and calculate metrics for all simulation combinations
    
    Args:
        consolidated_slo_stats: Dictionary of SLO stats DataFrames
        consolidated_worker_stats: Dictionary of worker stats DataFrames
        
    Returns:
        List of metrics dictionaries
    """
    metrics_data = []
    
    for (sim_type_name, agent_type_name), slo_stats_df in consolidated_slo_stats.items():
        worker_stats_df = consolidated_worker_stats[(sim_type_name, agent_type_name)]
        
        # Convert string names back to enums for plotting functions
        sim_type = getattr(SimulationType, sim_type_name.upper())
        agent_type = getattr(AgentType, agent_type_name.upper())
        
        # Create plots using the consolidated data
        plot_all_slo_stats(slo_stats_df, agent_type, sim_type)
        plot_all_worker_stats(worker_stats_df, agent_type, sim_type)
        
        # Calculate metrics using the consolidated data
        metrics = calculate_and_save_slo_metrics(slo_stats_df, agent_type_name, sim_type_name)
        
        metrics_row = metrics.copy()
        metrics_row['simulation_type'] = sim_type_name
        metrics_row['agent_type'] = agent_type_name
        metrics_data.append(metrics_row)
    
    # Create combined plots for all simulation types
    print("Creating combined agent comparison plots...")
    create_all_combined_plots(consolidated_slo_stats)
    
    return metrics_data


def create_agent_comparisons(metrics_data: list) -> pd.DataFrame:
    """
    Create agent performance comparisons from calculated metrics
    
    Args:
        metrics_data: List of metrics dictionaries
        
    Returns:
        DataFrame with agent comparison results
    """
    # Create multi-indexed DataFrame for metrics
    metrics_df = create_all_metrics_dataframe(metrics_data)

    # For each simulation type, compare the calculated values between agents
    comparison_df = compare_agent_metrics(metrics_df)
    save_comparison_results(comparison_df)
    
    return comparison_df


def generate_latex_tables(comparison_df: pd.DataFrame) -> None:
    """
    Generate LaTeX tables from agent comparison results
    
    Args:
        comparison_df: DataFrame with agent comparison results
    """
    generate_and_save_latex_tables(comparison_df)


def evaluate():
    """Run evaluation pipeline for all simulation data"""
    sns.set_theme(context='paper', style='white', palette='colorblind')
    
    # Step 1: Load all simulation data
    slo_stats_data, worker_stats_data = load_all_simulation_data()
    
    # Step 2: Create simulation data dictionaries for easy access
    consolidated_slo_stats, consolidated_worker_stats = create_simulation_data_dictionaries(
        slo_stats_data, worker_stats_data
    )
    
    # Step 3: Create multi-indexed simulation DataFrames (no longer saves unnecessary CSV files)
    big_slo_stats_df, big_worker_stats_df = create_and_save_simulation_dataframes(
        consolidated_slo_stats, consolidated_worker_stats
    )
    
    # Step 4: Generate plots and calculate metrics
    metrics_data = generate_plots_and_calculate_metrics(consolidated_slo_stats, consolidated_worker_stats)
    
    # Step 5: Create agent performance comparisons
    comparison_df = create_agent_comparisons(metrics_data)
    
    # Step 6: Generate LaTeX tables
    generate_latex_tables(comparison_df)
    
    # Return the consolidated DataFrames for potential future use
    return consolidated_slo_stats, consolidated_worker_stats
    

if __name__ == "__main__":
    evaluate()
