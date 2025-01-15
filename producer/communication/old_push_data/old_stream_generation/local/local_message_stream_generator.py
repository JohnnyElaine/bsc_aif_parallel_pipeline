import logging
import time
import msgpack
import msgpack_numpy as mnp

from multiprocessing.connection import Listener

import numpy as np

from producer.communication.old_push_data.old_stream_generation.node_info.node_info import NodeInfo
from producer.communication.old_push_data.old_stream_generation.stream_generator import StreamGenerator

log = logging.getLogger("producer")
mnp.patch()

class LocalMessageStreamGenerator(StreamGenerator):
    def __init__(self, port: int, video_path, nodes: list[NodeInfo]):
        super().__init__(video_path, nodes)
        self._listener = Listener(('localhost', port))

    def run(self):
        if not self._video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self._video.path}')

        self._wait_for_nodes_to_connect()

        try:
            self._stream_video()
        except ConnectionAbortedError:
            log.exception(f'Connection with a node failed')
        finally:
            self.stop()

    def stop(self):
        """
        Stops the video stream, releases the video capture
        :return:
        """
        log.info('stopping message-stream-generator')
        self._video.release()
        self._disconnect_nodes()

    def _wait_for_nodes_to_connect(self):
        num_connected_nodes = 0

        while num_connected_nodes < len(self.nodes):
            conn = self._listener.accept()
            self._add_connection(conn, num_connected_nodes)

            num_connected_nodes += 1

    def _add_connection(self, connection, index):
        # TODO actually associated incoming connection to existing node
        self.nodes[index].data_stream_connection = connection

    def _disconnect_nodes(self):
        for node in self.nodes:
            if (node is not None) and (not node.data_stream_connection.closed):
                node.data_stream_connection.close()

    def _stream_video(self):
        try:
            while self._video.is_opened():
                ok = self._iteration()
                if not ok:
                    break

        finally:
            self.stop()

    def _iteration(self):
        iteration_start_time = time.perf_counter()

        ret, frame, frame_index = self._video.read_frame()
        if not ret:
            log.error("End of video stream or error reading frame.")
            return False

        self._send_frame(frame, frame_index)
        self._enforce_target_fps(iteration_start_time, self._target_frame_time)

        return True

    def _send_frame(self, frame: np.ndarray, frame_index: int):
        target_node_index = self._determine_target_node_index()

        conn = self.nodes[target_node_index].data_stream_connection
        message = msgpack.packb({
            "frame_index": frame_index,
            "frame": frame,
        })

        conn.send_bytes(message)

        self._increment_current_target()




