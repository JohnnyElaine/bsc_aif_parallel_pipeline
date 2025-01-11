import socket
import msgpack
import numpy as np

from aif_edge_worker.communication.data.stream_receiver.stream_receiver import StreamReceiver


class UDPStreamReceiver(StreamReceiver):
    MAX_UDP_PACKET_SIZE = 32768

    def __init__(self, ip, port):
        super().__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((ip, port))
        self.frames = {}
        self.current_frame_data = {}

    def start(self):
        print(f"Listening for UDP packets on {self.socket.getsockname()}")

        while True:
            data, addr = self.socket.recvfrom(self.MAX_UDP_PACKET_SIZE)
            self._process_packet(data, addr)

    def _process_packet(self, data, addr):
        # Unpack the received message
        packet = msgpack.unpackb(data)
        frame_index = packet["frame_index"]
        sequence = packet["sequence"]
        is_last = packet["is_last"]
        chunk = packet["chunk"]

        # Store the chunk in the current frame data
        if frame_index not in self.current_frame_data:
            self.current_frame_data[frame_index] = {}
        self.current_frame_data[frame_index][sequence] = chunk

        # If this is the last chunk, assemble the frame
        if is_last:
            self._assemble_frame(frame_index)

    def _assemble_frame(self, frame_index):
        # Reconstruct the frame data in order
        chunks = self.current_frame_data[frame_index]
        sorted_chunks = [chunks[i] for i in sorted(chunks.keys())]
        frame_data = b"".join(sorted_chunks)

        # Decode the frame using msgpack
        frame = msgpack.unpackb(frame_data)["frame"]
        np_frame = np.array(frame)  # Convert back to numpy array if needed

        print(f"Frame {frame_index} received and assembled.")
        del self.current_frame_data[frame_index]  # Cleanup memory