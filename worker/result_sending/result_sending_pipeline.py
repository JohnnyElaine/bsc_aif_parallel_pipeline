from multiprocessing import Process, Pipe
from queue import Queue

import packages.logging as logging
from worker.result_sending.pipe_result_receiver.pipe_result_receiver import PipeResultReceiver
from worker.result_sending.result_sender.result_sender import ResultSender


class ResultSendingPipeline(Process):
    def __init__(self, collector_ip: str, collector_port: int, result_pipe_recv_end: Pipe):
        super().__init__()
        self._collector_ip = collector_ip
        self._collector_port = collector_port
        self._result_pipe = result_pipe_recv_end

    def run(self):
        log = logging.setup_logging('result_sending')
        log.debug('starting result-sending-pipeline')
        result_queue = Queue()
        result_receiver = PipeResultReceiver(result_queue, self._result_pipe)
        result_sender = ResultSender(result_queue, self._collector_ip, self._collector_port)

        result_receiver.start()
        result_sender.start()

        result_receiver.join()
        result_sender.join()

        log.debug('stopped result-sending-pipeline')