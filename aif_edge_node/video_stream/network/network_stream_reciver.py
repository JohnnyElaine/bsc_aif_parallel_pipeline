import socket
import msgpack
import numpy as np

from aif_edge_node.video_stream.stream_receiver import StreamReceiver


class NetworkStreamReceiver(StreamReceiver):
    def __init__(self):
        pass

    def start(self):
        """
        Start the UDP server to receive frames.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        print(f"Node-{self.id} listening on {self.host}:{self.port}")

        while True:
            data, _ = sock.recvfrom(65536)
            frame_info = msgpack.unpackb(data)
            frame_id = frame_info["frame_id"]
            frame = np.array(frame_info["frame"], dtype=np.uint8)

            # Process the received frame
            self._process_frame(frame, frame_id)

    def stop(self):
        pass

    def _process_frame(self, frame, frame_id):
        pass