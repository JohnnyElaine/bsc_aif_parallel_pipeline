import logging
import socket
import time
import numpy as np
import msgpack
import msgpack_numpy as mnp

from controller.communication.data.stream_generation.network.network_stream_generator import NetworkStreamGenerator
from controller.communication.data.stream_generation.node_info.node_info import NodeInfo
from controller.communication.data.stream_generation.video import Video

log = logging.getLogger("controller")

class UDPStreamGenerator(NetworkStreamGenerator):
    MAX_UDP_PACKET_SIZE = 32768

    def __init__(self, nodes: list[NodeInfo], video_path):
        """
        Initialize the NetworkStreamGenerator.

        :param nodes: List of tuples representing edge nodes (IP, port).
        :param video_path: Path to the video file.
        """
        super().__init__(nodes, video_path)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # bind socket

    def run(self):
        """
        Start reading the video and streaming frames to edge nodes.
        """
        mnp.patch() # enable numpy message packing
        video = Video(self.video_path)

        if not video.is_opened():
            raise ValueError(f"Error: Cannot open video file {self.video_path}")

        # Get the video's frames per second (fps)
        target_frame_interval = 1 / video.fps

        try:
            while video.is_opened():
                ok = self._iteration(video, target_frame_interval)
                if not ok:
                    break

        finally:
            video.release()
            self.socket.close()

    def _iteration(self, video: Video, target_frame_interval: float):
        iteration_start_time = time.perf_counter()
        ret, frame, frame_index = video.read_frame() # frame = ndarray
        if not ret:
            log.error("End of video reached")
            return False

        target_node_index = self._determine_target_node_index()
        target_node = self.nodes[target_node_index]


        self._send_frame(frame, frame_index, target_node.ip, target_node.port)

        self._enforce_target_fps(iteration_start_time, target_frame_interval)

        return True

    def _send_frame(self, frame: np.ndarray, frame_index: int, ip: str, port: int):
        chunks = self._chunk_data(frame.tobytes())

        for seq_num, chunk in enumerate(chunks):
            message = msgpack.packb({
                "sequence": seq_num,
                "frame_index": frame_index,
                "is_last": seq_num == len(chunks) - 1,
                "chunk": chunk,
            })
            self.socket.sendto(message, (ip, port))

    def _chunk_data(self, data: bytes) :
        """Split data into chunks of size MAX_UDP_PACKET_SIZE."""
        return [data[i:i + self.MAX_UDP_PACKET_SIZE] for i in range(0, len(data), self.MAX_UDP_PACKET_SIZE)]