import logging
import threading
import socket
from multiprocessing import Process

import cv2
import msgpack

from controller.stream_generation.local.local_message_stream_generator import LocalMessageStreamGenerator

log = logging.getLogger("controller")

class Controller:
    def __init__(self, port: int, video_path, nodes):
        """
        Initialize the coordinator with the video path and edge node information.
        :param video_path: Path to the input video file.
        :param nodes: List of (node_id, (host, port)) tuples representing edge nodes.
        """
        self._port = port
        self._video_path = video_path
        self._nodes = nodes
        self._offloading_map = {}

    def start(self):
        log.debug("starting controller")
        process = Process(target=self._run)
        process.start()
        return process

    def _run(self):
        stream_generator = LocalMessageStreamGenerator(self._port, self._video_path, self._nodes)
        stream_generator.start()

    def request_offloading(self, node_id, percentage, targets):
        """
        Record an offloading request.
        :param node_id: ID of the node requesting offloading.
        :param percentage: Percentage of tasks to offload (e.g., 50% = 0.5).
        :param targets: List of target nodes for offloading.
        """
        self._offloading_map[node_id] = {"percentage": percentage, "targets": targets}

    def assign_frame(self, frame, frame_id):
        """
        Determine the destination node for a given frame.
        """
        responsible_node_id = frame_id % len(self._nodes)

        # Check for offloading requests
        if responsible_node_id in self._offloading_map:
            offloading_info = self._offloading_map[responsible_node_id]
            percentage = offloading_info["percentage"]
            targets = offloading_info["targets"]

            if frame_id % 2 == 0 and percentage >= 0.5:  # Example: Offload 50%
                target_node = targets[frame_id % len(targets)]
                return target_node

        return self._nodes[responsible_node_id]

    def stream_video(self):
        """
        Stream video frames to edge nodes using UDP.
        """
        cap = cv2.VideoCapture(self._video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            # Determine the target node for this frame
            target_node = self.assign_frame(frame, frame_id)

            # Send frame to the target node
            threading.Thread(
                target=self.send_frame, args=(frame, frame_id, target_node[1])
            ).start()

            frame_id += 1

        cap.release()

    def send_frame(self, frame, frame_id, address):
        """
        Serialize and send a frame to the specified node using UDP.
        :param frame: The video frame (numpy array).
        :param frame_id: The ID of the frame.
        :param address: Tuple (host, port) of the target node.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = msgpack.packb({"frame_id": frame_id, "frame": frame.tolist()})
        sock.sendto(data, address)
        sock.close()

