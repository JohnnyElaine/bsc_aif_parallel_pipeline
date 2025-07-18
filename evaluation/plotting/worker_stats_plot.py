import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_all_worker_stats(worker_stats: pd.DataFrame, agent_type_name: str, sim_type_name: str, output_dir: str = "out/img"):
    """Plot all worker statistics and save to files"""
    # Create output directory if it doesn't exist
    dir_path = os.path.join(output_dir, f'{sim_type_name}_sim')
    os.makedirs(dir_path, exist_ok=True)
    
    # Construct filepath
    task_distribution_filepath = os.path.join(dir_path, f'{agent_type_name}_task_distribution_pie.png')
    
    plot_task_distribution_pie(worker_stats, task_distribution_filepath)

def plot_task_distribution_pie(worker_stats: pd.DataFrame, filepath: str = None):
    """Plot task distribution among workers as a pie chart and save to file"""
    plt.figure(figsize=(10, 8))
    
    # Create labels with worker ID and task count
    labels = [f'Worker {idx}\n({tasks} tasks)' for idx, tasks in zip(worker_stats.index, worker_stats['num_requested_tasks'])]
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(worker_stats['num_requested_tasks'],
                                      labels=labels,
                                      autopct='%1.1f%%',
                                      colors=sns.color_palette('Set3', len(worker_stats)),
                                      startangle=90,
                                      textprops={'fontsize': 10})
    
    # Enhance the appearance
    plt.title('Task Distribution Among Workers', fontsize=16, fontweight='bold', pad=20)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    
    # Save the plot
    if filepath:
        save_plot(filepath)
    else:
        plt.show()


def plot_task_distribution_bar(worker_stats: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    workers = worker_stats.sort_values('num_requested_tasks', ascending=False)

    sns.barplot(data=workers,
                x=workers.index,
                y='num_requested_tasks',
                hue=workers.index,  # Add this
                palette='viridis',
                legend=False)  # Add this

    plt.title('Tasks Requested per Worker')
    plt.xlabel('Worker ID')
    plt.ylabel('Number of Tasks')
    plt.show()

def save_plot(filepath):
    """Save plot to file with consistent formatting"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()