from abc import ABC

from controller.communication.data.stream_generation.node_info.node_info import NodeInfo
from controller.communication.data.stream_generation.stream_generator import StreamGenerator


class NetworkStreamGenerator(StreamGenerator, ABC):
    def __init__(self, port: int, video_path, nodes: list[NodeInfo]):
        super().__init__(video_path, nodes)
        self._port = port