import time
import logging

import msgpack
import msgpack_numpy as mnp
import cv2 as cv

from multiprocessing.connection import Client

from aif_edge_node.video_stream.stream_receiver import StreamReceiver

mnp.patch()
log = logging.getLogger("aif_edge_node")


class LocalMessageStreamReceiver(StreamReceiver):
    def __init__(self, port: int):
        self._is_running = True
        self._connection = Client(('localhost', port))

    def start(self):
        while self._is_running:
            try:
                ok = self._iteration()
                if not ok:
                    break
            except EOFError:
                print("Producer disconnected. Node exiting.")
                break

    def _iteration(self):
        message = self._connection.recv_bytes()
        data = msgpack.unpackb(message)

        frame_index = data["frame_index"]
        frame = data["frame"]

        # Display the frame (optional)
        cv.imshow("Node", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False

        return True
    def stop(self):
        self._is_running = False