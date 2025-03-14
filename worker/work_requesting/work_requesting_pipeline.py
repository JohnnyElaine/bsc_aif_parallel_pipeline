from multiprocessing import Process, Pipe
from queue import Queue

from worker.communication.channel.request_channel import RequestChannel
from worker.work_requesting.work_requester.network.zmq_work_requester import ZmqWorkRequester
from worker.work_requesting.pipe_task_sender.pipe_task_sender import PipeTaskSender

class WorkRequestingPipeline(Process):
    def __init__(self, channel: RequestChannel, task_pipe_sending_end: Pipe):
        super().__init__()
        self._channel = channel
        self._task_pipe = task_pipe_sending_end

    def run(self):
        # create shared (frame buffer) queue for work_requester & pipe sender
        task_queue = Queue()
        work_requester = ZmqWorkRequester(task_queue, self._channel)
        pipe_sender = PipeTaskSender(task_queue, self._task_pipe)

        work_requester.start()
        pipe_sender.start()

        work_requester.join()
        pipe_sender.join()