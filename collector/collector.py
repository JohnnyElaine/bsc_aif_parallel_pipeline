from multiprocessing import Process

import packages.logging as logging
from collector.collector_config import CollectorConfig
from collector.communication.result_collector import ResultCollector


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

        log.info("starting collector")
        
        result_collector = ResultCollector(self.config.port)
        
        # call run to avoid killing parent thread
        result_collector.run()