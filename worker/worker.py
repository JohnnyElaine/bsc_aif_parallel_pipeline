from multiprocessing import Process, Pipe, Event
from queue import Queue

import packages.logging as logging
from worker.communication.channel.request_channel import RequestChannel
from worker.communication.work_requester.network.zmq_work_requester import ZmqWorkRequester
from worker.computation.task_handling.task_handler import TaskHandler
from worker.communication.pipe_sender.pipe_sender import PipeSender
from worker.worker_config import WorkerConfig


class Worker(Process):
    def __init__(self, config: WorkerConfig):
        super().__init__()
        self.config = config

    def run(self):
        log = logging.setup_logging('worker')

        log.info(f"starting worker-{self.config.identity}")
        
        request_channel = RequestChannel(self.config.producer_ip, self.config.producer_port, self.config.identity)
        request_channel.connect()
        log.debug(f"established connection to producer-{request_channel}")

        work_config = self._register_at_producer(request_channel)

        if work_config is None:
            log.info(f'failed to register at producer {request_channel}')
            return

        log.info(f'registered at producer {request_channel}. Received config {work_config}')

        # Pipe for IPC
        pipe_receiving_end, pipe_sending_end = Pipe(False)
        log.debug('created pipe for IPC (Main Process -> Task Processor)')

        task_handler_ready = Event()
        task_handler = TaskHandler(self.config.identity, pipe_receiving_end, work_config, task_handler_ready)

        # create shared (frame buffer) queue for work_requester & pipe sender
        task_queue = Queue()
        work_requester = ZmqWorkRequester(task_queue, request_channel)
        pipe_sender = PipeSender(task_queue, pipe_sending_end)

        task_handler.start()
        pipe_sender.start()

        # only start requesting work, when the task_handler is ready
        task_handler_ready.wait()
        work_requester.start()

        work_requester.join()
        pipe_sender.join()
        task_handler.join()

    def _register_at_producer(self, control_channel: RequestChannel):
        return control_channel.register()