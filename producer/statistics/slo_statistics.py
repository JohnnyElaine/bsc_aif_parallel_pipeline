import pandas as pd


class SloStatistics:
    def __init__(self):
        # Raw metrics
        self.queue_size = []
        self.memory_usage = []

        # SLO ratios
        self.queue_size_ratio = []
        self.memory_usage_ratio = []

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert statistics to a pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing SLO statistics
        """
        data = {
            'queue_size': self.queue_size,
            'memory_usage': self.memory_usage,
            'queue_size_ratio': self.queue_size_ratio,
            'memory_usage_ratio': self.memory_usage_ratio,
        }

        return pd.DataFrame(data)

    def __str__(self):
        return (
            f"SloStatistics:\n"
            f"  queue_size (len={len(self.queue_size)}): {self.queue_size[:5]}{'...' if len(self.queue_size) > 5 else ''}\n"
            f"  memory_usage (len={len(self.memory_usage)}): {self.memory_usage[:5]}{'...' if len(self.memory_usage) > 5 else ''}\n"
            f"  queue_size_ratio (len={len(self.queue_size_ratio)}): {self.queue_size_ratio[:5]}{'...' if len(self.queue_size_ratio) > 5 else ''}\n"
            f"  memory_usage_ratio (len={len(self.memory_usage_ratio)}): {self.memory_usage_ratio[:5]}{'...' if len(self.memory_usage_ratio) > 5 else ''}"
        )

    def __repr__(self):
        return (
            f"SloStatistics:\n"
            f"  queue_size (len={len(self.queue_size)}): {self.queue_size[:5]}{'...' if len(self.queue_size) > 5 else ''}\n"
            f"  memory_usage (len={len(self.memory_usage)}): {self.memory_usage[:5]}{'...' if len(self.memory_usage) > 5 else ''}\n"
            f"  queue_size_ratio (len={len(self.queue_size_ratio)}): {self.queue_size_ratio[:5]}{'...' if len(self.queue_size_ratio) > 5 else ''}\n"
            f"  memory_usage_ratio (len={len(self.memory_usage_ratio)}): {self.memory_usage_ratio[:5]}{'...' if len(self.memory_usage_ratio) > 5 else ''}"
        )