import pandas as pd


class SloStatistics:
    def __init__(self):
        # Quality config (percentage of max capacity (0-1), e.g. fps = 1.0 if streams runs at source fps)
        self.fps_capacity = []
        self.resolution_capacity = []
        self.work_load_capacity = []

        # Raw metrics
        self.queue_size = []
        self.memory_usage = []

        # SLO ratios
        self.queue_size_slo_ratio = []
        self.memory_usage_slo_ratio = []

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert statistics to a pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing SLO statistics
        """
        data = {
            'fps_capacity': self.fps_capacity,
            'resolution_capacity': self.resolution_capacity,
            'work_load_capacity': self.work_load_capacity,
            'queue_size': self.queue_size,
            'memory_usage': self.memory_usage,
            'queue_size_slo_ratio': self.queue_size_slo_ratio,
            'memory_usage_slo_ratio': self.memory_usage_slo_ratio
        }

        return pd.DataFrame(data)