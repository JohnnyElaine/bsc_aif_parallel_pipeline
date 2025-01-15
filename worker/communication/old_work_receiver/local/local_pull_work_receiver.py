import logging

from queue import Queue

import msgpack
import msgpack_numpy as mnp

from multiprocessing.connection import Client

from worker.communication.old_work_receiver.stream_receiver import WorkReceiver

mnp.patch()

log = logging.getLogger("worker")


class LocalWorkReceiver(WorkReceiver):
    def __init__(self, port: int, shared_queue: Queue):
        super().__init__(shared_queue)
        self._connection = Client(('localhost', port))
        self._queue = shared_queue
        self._is_running = False

    def run(self):
        log.debug("starting stream-receiver")
        self._is_running = True
        while self._is_running:
            try:
                ok = self._iteration()
                if not ok:
                    break
            except EOFError:
                print("Producer disconnected. Node exiting.")
                break

    def stop(self):
        log.info("stopping stream-receiver")

    def _iteration(self):
        """
        :return: True if the iteration was successful. False otherwise.
        """
        data = self._receive_message()
        log.debug(f'received frame: {data['frame_index']}, frame-buffer size: {self._queue.qsize()}')

        if data is None:
            return False

        self._queue.put(data)

        return True

    def _receive_message(self):
        message = self._connection.recv_bytes()
        #frame_index = old_push_data["frame_index"]
        #frame = old_push_data["frame"]
        return msgpack.unpackb(message)
