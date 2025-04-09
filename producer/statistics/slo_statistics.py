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