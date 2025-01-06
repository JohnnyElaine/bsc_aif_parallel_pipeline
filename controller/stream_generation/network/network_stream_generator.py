from abc import ABC

from controller.stream_generation.node_info.node_info import NodeInfo
from controller.stream_generation.stream_generator import StreamGenerator


class NetworkStreamGenerator(StreamGenerator, ABC):
    def __init__(self, nodes: list[NodeInfo], video_path):
        super().__init__(video_path)
        self.nodes = nodes
        self._curr_target = 0
        self._curr_offload_target = 0

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