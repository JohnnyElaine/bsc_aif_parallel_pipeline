from queue import Queue

from worker.communication.old_work_receiver.stream_receiver import WorkReceiver


class ZeroMQWorkReceiver(WorkReceiver):
    def __init__(self, shared_queue: Queue, port: int):
        super().__init__(shared_queue)
        self._port = port