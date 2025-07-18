from multiprocessing import Process, Pipe, Value
from queue import Queue

import packages.logging as logging
from worker.communication.channel.request_channel import RequestChannel
from worker.work_requesting.pipe_task_sender.pipe_task_sender import PipeTaskSender
from worker.work_requesting.work_requester.network.zmq_work_requester import ZmqWorkRequester


class WorkRequestingPipeline(Process):
    def __init__(self, channel: RequestChannel, task_pipe_sending_end: Pipe, shared_processing_time: Value, outage_config=None):
        super().__init__()
        self._channel = channel
        self._task_pipe = task_pipe_sending_end
        self._latest_processing_time = shared_processing_time
        self._outage_config = outage_config

    def run(self):
        log = logging.setup_logging('work_requesting')

        log.debug('starting work-requesting-pipeline')
        # create shared (frame buffer) queue for work_requester & pipe sender
        task_todo_queue = Queue()
        work_requester = ZmqWorkRequester(task_todo_queue, self._channel, self._latest_processing_time, outage_config=self._outage_config)
        pipe_sender = PipeTaskSender(task_todo_queue, self._task_pipe)

        work_requester.start()
        pipe_sender.start()

        work_requester.join()
        pipe_sender.join()

        log.debug('stopped work-requesting-pipeline')