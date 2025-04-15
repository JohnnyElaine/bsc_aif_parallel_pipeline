import matplotlib.pyplot as plt
import seaborn as sns

def plot_all(slo_stats):
    plot_slo_ratios_over_time(slo_stats)
    plot_quality_metrics(slo_stats)

def plot_slo_ratios_over_time(slo_stats):
    """Plot both SLO ratios over time with critical threshold"""
    slo_stats = slo_stats.reset_index()

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=slo_stats, x='index', y='queue_size_slo_ratio',
                 label='Queue Size Ratio', color='blue', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='memory_usage_slo_ratio',
                 label='Memory Usage Ratio', color='red', linewidth=2)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=2,
                label='SLO Fulfillment Threshold')
    plt.title('SLO Ratios Over Time', fontsize=16)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Ratio Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_quality_metrics(slo_stats):
    """Plot capacity metrics over time"""
    slo_stats = slo_stats.reset_index()

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=slo_stats, x='index', y='fps_capacity',
                 label='FPS', color='red', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='resolution_capacity',
                 label='Resolution', color='green', linewidth=2)
    sns.lineplot(data=slo_stats, x='index', y='work_load_capacity',
                 label='Work Load (Pixels)', color='blue', linewidth=2)

    plt.title('Quality Metrics Over Time', fontsize=16)
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Capacity Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
