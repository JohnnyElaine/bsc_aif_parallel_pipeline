import pandas as pd


class SloStatistics:
    def __init__(self):
        # Quality config (percentage of max capacity (0-1), e.g. fps = 1.0 if streams runs at source fps)
        self.fps_capacity = []
        self.resolution_capacity = []
        self.inference_quality = []

        # Raw metrics
        self.queue_size = []
        self.memory_usage = []
        self.avg_global_processing_time = []
        self.avg_worker_processing_time = {} # TODO add to dataframe

        # SLO values
        self.queue_size_slo_value = []
        self.memory_usage_slo_value = []
        self.avg_global_processing_time_slo_value = []
        self.avg_worker_processing_time_slo_value = []

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert statistics to a pandas DataFrame

        Returns:
            pd.DataFrame: DataFrame containing SLO statistics
        """
        data = {
            'fps_capacity': self.fps_capacity,
            'resolution_capacity': self.resolution_capacity,
            'work_load_capacity': self.inference_quality,
            'queue_size': self.queue_size,
            'memory_usage': self.memory_usage,
            'avg_global_processing_time': self.avg_global_processing_time,
            'queue_size_slo_value': self.queue_size_slo_value,
            'memory_usage_slo_value': self.memory_usage_slo_value,
            'avg_global_processing_time_slo_value': self.avg_global_processing_time_slo_value,
            'avg_worker_processing_time_slo_value': self.avg_worker_processing_time_slo_value
        }

        return pd.DataFrame(data)