import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType
from ..evaluation_utils import EvaluationUtils
from ..enums.directory_type import DirectoryType

def plot_all_worker_stats(worker_stats: pd.DataFrame, agent_type: AgentType, sim_type: SimulationType, output_dir: str = "out/img"):
    """Plot all worker statistics and save to files"""
    # Get plot filepath from EvaluationUtils
    task_distribution_filepath = EvaluationUtils.get_filepath(DirectoryType.IMG, sim_type, agent_type, "task_distribution_pie", "pdf")
    
    # Ensure directory exists
    EvaluationUtils.ensure_directory_exists(task_distribution_filepath)
    
    plot_task_distribution_pie(worker_stats, task_distribution_filepath)

def plot_task_distribution_pie(worker_stats: pd.DataFrame, filepath: str = None):
    """Plot task distribution among workers as a pie chart and save to file"""
    plt.figure(figsize=(8, 5))
    
    # Create labels with worker ID and task count
    labels = [f'Worker {idx}\n({tasks} tasks)' for idx, tasks in zip(worker_stats.index, worker_stats['num_requested_tasks'])]
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(worker_stats['num_requested_tasks'],
                                      labels=labels,
                                      autopct='%1.1f%%',
                                      colors=sns.color_palette('Set3', len(worker_stats)),
                                      startangle=90,
                                      textprops={'fontsize': 12})
    
    # Enhance the appearance
    plt.title('Task Distribution Among Workers', fontsize=18, fontweight='bold', pad=20)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    
    # Save the plot
    if filepath:
        save_plot(filepath)
    else:
        plt.show()


def plot_task_distribution_bar(worker_stats: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    workers = worker_stats.sort_values('num_requested_tasks', ascending=False)

    sns.barplot(data=workers,
                x=workers.index,
                y='num_requested_tasks',
                hue=workers.index,  # Add this
                palette='viridis',
                legend=False)  # Add this

    plt.title('Tasks Requested per Worker', fontsize=18)
    plt.xlabel('Worker ID', fontsize=14)
    plt.ylabel('Number of Tasks', fontsize=14)
    plt.show()

def save_plot(filepath):
    """Save plot to file with consistent formatting"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()