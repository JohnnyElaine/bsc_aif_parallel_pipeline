import logging
from threading import Thread, Event

from collector.communication.channel.pull_channel import PullChannel

log = logging.getLogger("collector")


class ResultCollector(Thread):
    def __init__(self, port: int):
        super().__init__()
        self._channel = PullChannel(port, )
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
        log.info("stopping result-collector")
        self._is_running = False
        self._channel.close()
        
    def _iteration(self):
        results = self._channel.get_results()
        # TODO: do something with result
    
        return True
        