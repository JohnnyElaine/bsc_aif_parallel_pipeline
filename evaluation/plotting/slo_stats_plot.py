import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_all_slo_stats(slo_stats: pd.DataFrame, agent_type_name: str, sim_type_name: str, output_dir: str = "out/img"):
    """Plot all SLO statistics and save to files"""
    # Create output directory if it doesn't exist
    dir_path = os.path.join(output_dir, f'{sim_type_name}_sim')
    os.makedirs(dir_path, exist_ok=True)
    
    # Construct filepaths
    slo_values_filepath = os.path.join(dir_path, f'{agent_type_name}_slo_values.png')
    quality_metrics_filepath = os.path.join(dir_path, f'{agent_type_name}_quality_metrics.png')
    
    plot_slo_values_over_time(slo_stats, slo_values_filepath)
    plot_quality_metrics(slo_stats, quality_metrics_filepath)

def plot_slo_values_over_time(slo_stats: pd.DataFrame, filepath: str = None):
    """Plot both SLO ratios over time with critical threshold"""
    slo_stats = slo_stats.reset_index()
    
    # Cap values at upper bound for display purposes
    upper_bound = 4
    slo_stats_capped = slo_stats.copy()
    slo_stats_capped['queue_size_slo_value'] = slo_stats_capped['queue_size_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['memory_usage_slo_value'] = slo_stats_capped['memory_usage_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['avg_global_processing_time_slo_value'] = slo_stats_capped['avg_global_processing_time_slo_value'].clip(upper=upper_bound)
    slo_stats_capped['avg_worker_processing_time_slo_value'] = slo_stats_capped['avg_worker_processing_time_slo_value'].clip(upper=upper_bound)

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
    plt.title(f'SLO Values Over Time', fontsize=16)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Ratio Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filepath:
        save_plot(filepath)
    else:
        plt.show()

def plot_quality_metrics(slo_stats, filepath: str = None):
    """Plot capacity metrics over time"""
    slo_stats = slo_stats.reset_index()

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=slo_stats, x='index', y='fps_capacity',
                 label='FPS', color='red', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='resolution_capacity',
                 label='Resolution', color='green', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='inference_quality_capacity',
                 label='Inference Quality', color='blue', linewidth=2)

    plt.title('Quality Metrics Over Time', fontsize=16)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Capacity Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        save_plot(filepath)
    else:
        plt.show()

def plot_queue_size_over_time(slo_stats):
    """Plot queue size over time"""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=slo_stats.reset_index(), x='index', y='queue_size', color='blue')
    plt.title('Queue Size Over Time', fontsize=14)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Queue Size', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_memory_usage_over_time(slo_stats):
    """Plot memory usage over time"""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=slo_stats.reset_index(), x='index', y='memory_usage', color='green')
    plt.title('Memory Usage Over Time', fontsize=14)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Memory Usage', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_queue_ratio_distribution(slo_stats):
    """Plot distribution of queue size ratios"""
    plt.figure(figsize=(10, 5))
    sns.histplot(data=slo_stats, x='queue_size_slo_ratio', bins=20, kde=True,
                 color='skyblue', edgecolor='white')
    plt.title('Distribution of Queue Size SLO Ratios', fontsize=14)
    plt.xlabel('Queue Size Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_memory_ratio_distribution(slo_stats):
    """Plot distribution of memory usage ratios"""
    plt.figure(figsize=(10, 5))
    sns.histplot(data=slo_stats, x='memory_usage_slo_ratio', bins=20, kde=True,
                 color='salmon', edgecolor='white')
    plt.title('Distribution of Memory Usage SLO Ratios', fontsize=14)
    plt.xlabel('Memory Usage Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_plot(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
