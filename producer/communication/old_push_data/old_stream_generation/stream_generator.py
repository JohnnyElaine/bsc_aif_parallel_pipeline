import time

from abc import ABC, abstractmethod

from producer.communication.old_push_data.old_stream_generation.node_info.node_info import NodeInfo
from producer.stream_generation.video import Video


class StreamGenerator(ABC):
    def __init__(self, video_path, nodes: list[NodeInfo]):
        self.nodes = nodes
        self._video = Video(video_path)
        self._target_frame_time = 1 / self._video.fps
        self._curr_target = 0
        self._curr_offload_target = 0

    @abstractmethod
    def run(self):
        pass

    def _determine_target_node_index(self):
        if not self.nodes[self._curr_target].is_offloading():
            return self._curr_target

        self._increment_offload_target()

        # prevent offloading back to the node that requested offloading
        if self._curr_offload_target == self._curr_target:
            self._increment_offload_target()

        return self._curr_offload_target

    def _increment_offload_target(self):
        self._curr_offload_target = (self._curr_offload_target + 1) % len(self.nodes)

    def _increment_current_target(self):
        self._curr_target = (self._curr_target + 1) % len(self.nodes)

    @staticmethod
    def _enforce_target_fps(iteration_start_time: float, target_frame_interval: float):
        iteration_duration = time.perf_counter() - iteration_start_time
        wait_time = max(target_frame_interval - iteration_duration, 0)
        if wait_time > 0:
            time.sleep(wait_time)