import logging
from threading import Thread
from queue import PriorityQueue

from collector.communication.channel.pull_channel import PullChannel

log = logging.getLogger('collector')


class ResultCollector(Thread):
    def __init__(self, port: int, result_queue: PriorityQueue):
        super().__init__()
        self._channel = PullChannel(port)
        self._result_queue = result_queue
        self._is_running = False
        
    def run(self):
        self._is_running = True
        self._channel.bind()
        log.debug(f'bound {self._channel}')
        
        while self._is_running:
            ok = self._iteration()      
            
            if not ok:
                self.stop()
                break
                
    def stop(self):
        log.info('stopping result-collector')
        self._is_running = False
        self._channel.close()
        
    def _iteration(self) -> bool:
        results = self._channel.get_results() # result = Task(id, type, data)

        for result in results:
            self._result_queue.put(result)

        return True
        