import matplotlib.pyplot as plt
import seaborn as sns

def plot_all(slo_stats):
    plot_queue_size_over_time(slo_stats)
    plot_memory_usage_over_time(slo_stats)
    plot_queue_ratio_distribution(slo_stats)
    plot_memory_ratio_distribution(slo_stats)
    plot_queue_vs_memory(slo_stats)
    plot_slo_ratios_over_time(slo_stats)
    plot_slo_ratio_boxplots(slo_stats)

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
    sns.histplot(data=slo_stats, x='queue_size_ratio', bins=20, kde=True,
                 color='skyblue', edgecolor='white')
    plt.title('Distribution of Queue Size Ratios', fontsize=14)
    plt.xlabel('Queue Size Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_memory_ratio_distribution(slo_stats):
    """Plot distribution of memory usage ratios"""
    plt.figure(figsize=(10, 5))
    sns.histplot(data=slo_stats, x='memory_usage_ratio', bins=20, kde=True,
                 color='salmon', edgecolor='white')
    plt.title('Distribution of Memory Usage Ratios', fontsize=14)
    plt.xlabel('Memory Usage Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_queue_vs_memory(slo_stats):
    """Plot queue size vs memory usage colored by queue ratio"""
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=slo_stats, x='queue_size', y='memory_usage',
                             hue='queue_size_ratio', palette='viridis',
                             size='queue_size_ratio', sizes=(20, 200))
    plt.title('Queue Size vs Memory Usage (Colored by Queue Ratio)', fontsize=14)
    plt.xlabel('Queue Size', fontsize=12)
    plt.ylabel('Memory Usage', fontsize=12)
    plt.legend(title='Queue Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_slo_ratios_over_time(slo_stats):
    """Plot both SLO ratios over time with critical threshold"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=slo_stats.reset_index(), x='index', y='queue_size_ratio',
                label='Queue Size Ratio', color='blue', linewidth=2)
    sns.lineplot(data=slo_stats.reset_index(), x='index', y='memory_usage_ratio',
                label='Memory Usage Ratio', color='red', linewidth=2)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2,
                label='Critical Threshold')
    plt.title('SLO Ratios Over Time', fontsize=16)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Ratio Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_slo_ratio_boxplots(slo_stats):
    """Plot boxplots of both SLO ratios"""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=slo_stats[['queue_size_ratio', 'memory_usage_ratio']],
               palette=['skyblue', 'salmon'], width=0.5)
    plt.title('Distribution of SLO Ratios', fontsize=14)
    plt.ylabel('Ratio Value', fontsize=12)
    plt.xticks([0, 1], ['Queue Size Ratio', 'Memory Usage Ratio'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()