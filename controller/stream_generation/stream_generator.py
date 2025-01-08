import time

from abc import ABC, abstractmethod

class StreamGenerator(ABC):
    def __init__(self, video_path):
        self.video_path = video_path

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def _enforce_target_fps(iteration_start_time: float, target_frame_interval: float):
        iteration_duration = time.perf_counter() - iteration_start_time
        wait_time = max(target_frame_interval - iteration_duration, 0)
        if wait_time > 0:
            time.sleep(wait_time)