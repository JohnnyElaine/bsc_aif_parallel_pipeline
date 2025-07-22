import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType
from ..evaluation_utils import EvaluationUtils
from ..enums.directory_type import DirectoryType

def plot_all_slo_stats(slo_stats: pd.DataFrame, agent_type: AgentType, sim_type: SimulationType, output_dir: str = "out/img"):
    """Plot all SLO statistics and save to files"""
    # Get plot filepaths from EvaluationUtils
    slo_values_filepath = EvaluationUtils.get_filepath(DirectoryType.IMG, sim_type, agent_type, "slo_values", "pdf")
    quality_metrics_filepath = EvaluationUtils.get_filepath(DirectoryType.IMG, sim_type, agent_type, "quality_metrics", "pdf")
    
    # Ensure directory exists
    EvaluationUtils.ensure_directory_exists(slo_values_filepath)
    
    _plot_slo_values_over_time(slo_stats, slo_values_filepath)
    _plot_quality_metrics(slo_stats, quality_metrics_filepath)

def _plot_slo_values_over_time(slo_stats: pd.DataFrame, filepath: str = None, title_prefix: str = None, 
                              create_figure: bool = True, fontsize: int = 12):
    """Plot both SLO ratios over time with critical threshold"""
    slo_stats = slo_stats.reset_index()
    
    # Cap values at upper bound for display purposes
    upper_bound = 4
    slo_stats_capped = slo_stats.copy()
    slo_stats_capped['queue_size_slo_value'] = slo_stats_capped['queue_size_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['memory_usage_slo_value'] = slo_stats_capped['memory_usage_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['avg_global_processing_time_slo_value'] = slo_stats_capped['avg_global_processing_time_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['avg_worker_processing_time_slo_value'] = slo_stats_capped['avg_worker_processing_time_slo_value'].clip(upper=upper_bound)

    if create_figure:
        plt.figure(figsize=(12, 6))

    sns.lineplot(data=slo_stats_capped, x='index', y='queue_size_slo_value',
                 label='Queue Size', color='blue', linewidth=2)
    sns.lineplot(data=slo_stats_capped, x='index', y='memory_usage_slo_value',
                 label='Memory Usage', color='red', linewidth=2)
    sns.lineplot(data=slo_stats_capped, x='index', y='avg_global_processing_time_slo_value',
                 label='Global Processing Time', color='green', linewidth=2)
    sns.lineplot(data=slo_stats_capped, x='index', y='avg_worker_processing_time_slo_value',
                 label='Worker Processing Time', color='magenta', linewidth=2)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=2,
                label='SLO Fulfillment Threshold')
    
    # Set title based on whether it's a subplot or standalone
    title = f'{title_prefix} - SLO Values Over Time' if title_prefix else 'SLO Values Over Time'
    title_fontsize = 14 if title_prefix else 16
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Ratio Value', fontsize=12)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    
    if create_figure:
        plt.tight_layout()

    if filepath and create_figure:
        save_plot(filepath)
    elif not filepath and create_figure:
        plt.show()

def _plot_quality_metrics(slo_stats, filepath: str = None, title_prefix: str = None, 
                         create_figure: bool = True, fontsize: int = 12):
    """Plot capacity metrics over time"""
    slo_stats = slo_stats.reset_index()

    if create_figure:
        plt.figure(figsize=(12, 6))

    sns.lineplot(data=slo_stats, x='index', y='fps_capacity',
                 label='FPS', color='red', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='resolution_capacity',
                 label='Resolution', color='green', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='inference_quality_capacity',
                 label='Inference Quality', color='blue', linewidth=2)

    # Set title based on whether it's a subplot or standalone
    title = f'{title_prefix} - Quality Metrics Over Time' if title_prefix else 'Quality Metrics Over Time'
    title_fontsize = 14 if title_prefix else 16
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Capacity Value', fontsize=12)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    
    if create_figure:
        plt.tight_layout()
    
    if filepath and create_figure:
        save_plot(filepath)
    elif not filepath and create_figure:
        plt.show()


def create_all_combined_plots(slo_stats_data: dict):
    """
    Create combined plots for all simulation types and both plot types
    
    Args:
        slo_stats_data: Dictionary with keys as (sim_type_name, agent_type_name) and DataFrames as values
    """
    simulation_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]
    plot_types = ['slo_values', 'quality_metrics']
    
    for sim_type in simulation_types:
        for plot_type in plot_types:
            plot_combined_agent_comparison(slo_stats_data, sim_type, plot_type)

def plot_combined_agent_comparison(slo_stats_data: dict, sim_type: SimulationType, plot_type: str):
    """
    Create combined figures showing both agents for a simulation type
    
    Args:
        slo_stats_data: Dictionary with keys as (sim_type_name, agent_type_name) and DataFrames as values
        sim_type: SimulationType enum
        plot_type: Either 'slo_values' or 'quality_metrics'
    """
    sim_type_name = sim_type.name.lower()
    
    # Get data for both agents
    ai_key = (sim_type_name, 'active_inference_relative_control')
    heuristic_key = (sim_type_name, 'heuristic')
    
    if ai_key not in slo_stats_data or heuristic_key not in slo_stats_data:
        print(f"Warning: Missing data for {sim_type_name} combined plot")
        return
    
    ai_data = slo_stats_data[ai_key]
    heuristic_data = slo_stats_data[heuristic_key]
    
    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot AI agent at the top
    plt.sca(ax1)
    if plot_type == 'slo_values':
        _plot_slo_values_over_time(ai_data, title_prefix="Active Inference Agent", 
                                 create_figure=False, fontsize=10)
    else:  # quality_metrics
        _plot_quality_metrics(ai_data, title_prefix="Active Inference Agent", 
                            create_figure=False, fontsize=10)
    
    # Plot Heuristic agent at the bottom
    plt.sca(ax2)
    if plot_type == 'slo_values':
        _plot_slo_values_over_time(heuristic_data, title_prefix="Heuristic Agent", 
                                 create_figure=False, fontsize=10)
    else:  # quality_metrics
        _plot_quality_metrics(heuristic_data, title_prefix="Heuristic Agent", 
                            create_figure=False, fontsize=10)
    
    plt.tight_layout()
    
    # Save the combined plot
    filename = f"combined_{plot_type}"
    filepath = EvaluationUtils.get_filepath(DirectoryType.IMG, sim_type, None, filename, "pdf")
    EvaluationUtils.ensure_directory_exists(filepath)
    
    save_plot(filepath)

def save_plot(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
