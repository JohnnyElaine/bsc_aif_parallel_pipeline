from abc import ABC

from producer.communication.old_push_data.old_stream_generation.node_info.node_info import NodeInfo
from producer.communication.old_push_data.old_stream_generation.stream_generator import StreamGenerator


class NetworkStreamGenerator(StreamGenerator, ABC):
    def __init__(self, port: int, video_path, nodes: list[NodeInfo]):
        super().__init__(video_path, nodes)
        self._port = port