from queue import Queue

from aif_edge_worker.stream_receiver.stream_receiver import StreamReceiver


class ZeroMQStreamReceiver(StreamReceiver):
    def __init__(self, shared_queue: Queue, port: int):
        super().__init__(shared_queue)
        self._port = port