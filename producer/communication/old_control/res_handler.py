import logging
from threading import Thread

from producer.communication.old_control.control_channel.control_channel import ControlChannel

log = logging.getLogger("producer")


class ResHandler(Thread):
    def __init__(self, port: int):
        super().__init__()
        self._control_channel = ControlChannel(port)
        self._is_running = False

    def run(self):
        self._is_running = True
        while self._is_running:
            req = self._control_channel.get_req()
            # TODO handle request and send answer

    def stop(self):
        log.info('stopping res-handler')
        self._is_running = False