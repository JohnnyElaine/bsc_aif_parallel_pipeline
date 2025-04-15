import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_all_worker_stats(worker_stats: pd.DataFrame):
    plot_task_distribution_pie(worker_stats)
    plot_task_distribution_bar(worker_stats)

def plot_task_distribution_pie(worker_stats: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    plt.pie(worker_stats['num_requested_tasks'],
            labels=worker_stats.index,
            autopct='%1.1f%%',
            colors=sns.color_palette('pastel'),
            startangle=90)
    plt.title('Task Distribution Among Workers', fontsize=14)
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


def plot_old(worker_stats: pd.DataFrame):
    plt.figure(figsize=(12, 5))

    # Plot 1: Worker task distribution
    plt.subplot(1, 2, 1)
    workers = worker_stats.sort_values('num_requested_tasks', ascending=False)
    sns.barplot(data=workers, x=workers.index, y='num_requested_tasks', palette='viridis')
    plt.title('Tasks Requested per Worker')
    plt.xlabel('Worker ID')
    plt.ylabel('Number of Tasks')
    plt.xticks(rotation=45)

    # Plot 2: Registration time analysis
    plt.subplot(1, 2, 2)
    workers['registration_human'] = pd.to_datetime(workers['registration_time'], unit='s')
    sns.scatterplot(data=workers, x='registration_human', y='num_requested_tasks',
                    hue=workers.index, s=200)
    plt.title('Registration Time vs Task Count')
    plt.xlabel('Registration Time')
    plt.ylabel('Tasks Requested')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()