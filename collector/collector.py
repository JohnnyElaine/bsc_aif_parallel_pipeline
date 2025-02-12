from multiprocessing import Process
from queue import Queue, PriorityQueue

import packages.logging as logging
from collector.collector_config import CollectorConfig
from collector.output_viewer.output_viewer import OutputViewer
from collector.result_arrangement.result_arranger import ResultArranger
from collector.result_collection.result_collector import ResultCollector


class Collector(Process):
    def __init__(self, config: CollectorConfig):
        """
        Initialize the coordinator with the video path and edge node information.
        :param config: collector config
        """
        super().__init__()
        self.config = config

    def run(self):
        log = logging.setup_logging('collector')

        log.info('starting collector')

        result_queue = PriorityQueue()
        output_queue = Queue()

        result_collector = ResultCollector(self.config.port, result_queue)
        result_arranger = ResultArranger(result_queue, output_queue)
        output_viewer = OutputViewer(output_queue)

        result_arranger.start()
        result_collector.start()
        output_viewer.start()

        result_collector.join()
        result_arranger.join()
        output_viewer.join()