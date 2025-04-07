import logging
from threading import Thread
from queue import Queue

from packages.data.types.signal_type import SignalType
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
        log.debug('starting pipe-result-receiver')
        self._is_running = True
        self._channel.connect()

        while self._is_running:
            result = self._queue.get() # LocalMessage Dataclass

            if result.type == SignalType.END:
                self._channel.send_info(dict(type=RepType.END))
                self._is_running = False
                break

            self._channel.send_results([result]) # TODO maybe send more results at once if possible

        self._channel.close()

    def stop(self):
        self._is_running = False
        self._channel.close()
        log.info('stopped pipe-result-receiver')