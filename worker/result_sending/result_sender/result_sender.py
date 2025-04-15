import logging
from queue import Queue
from threading import Thread

from zmq import ZMQError

from packages.data.local_messages.task_type import TaskType
from packages.network_messages import RepType
from worker.communication.channel.push_channel import PushChannel

log = logging.getLogger('result_sending')


class ResultSender(Thread):
    def __init__(self, result_queue: Queue, collector_ip: str, collector_port: int):
        super().__init__()
        self._queue = result_queue
        self._channel = PushChannel(collector_ip, collector_port)
        self._is_running = False

    def run(self):
        log.debug('starting result-sender')
        self._is_running = True
        self._channel.connect()

        while self._is_running:
            result = self._queue.get() # LocalMessage Dataclass

            if result.type == TaskType.END:
                try:
                    self._channel.send_info(dict(type=RepType.END))
                    log.debug('sent END message to collector')
                except ZMQError as e:
                    log.warning(f'Failed to send END message: {e}')
                finally:
                    self._is_running = False
                    break

            self._channel.send_results([result]) # TODO maybe send more results at once if possible

        self._channel.close()
        log.debug('stopped result-sender')