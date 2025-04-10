import logging
from threading import Thread

import numpy as np

from collector.communication.channel.pull_channel import PullChannel
from collector.constants.constants import END_TASK_ID
from collector.datastructures.blocking_dict import BlockingDict
from packages.data.local_messages.task import Task
from packages.data.types.task_type import TaskType
from packages.network_messages import RepType

log = logging.getLogger('collector')


class ResultCollector(Thread):
    def __init__(self, port: int, result_queue: BlockingDict):
        super().__init__()
        self._channel = PullChannel(port)
        self._result_dict = result_queue
        self._is_running = False

    def run(self):
        self._is_running = True
        self._channel.bind()
        log.debug(f'bound {self._channel}')

        ok = True
        while self._is_running and ok:
            ok = self._iteration()

        self._channel.close()
        log.debug('stopped result-collector')

    def stop(self):
        log.info('stopping result-collector')
        self._is_running = False
        self._channel.close()
        
    def _iteration(self) -> bool:
        info, results = self._channel.get_results() # info = dict, results = list of Task()

        match info['type']:
            case RepType.END:
                # notify the ResultMapper that it should also stop
                self._result_dict[END_TASK_ID] = Task(TaskType.END, END_TASK_ID, np.empty(0))
                return False
            case _:
                for result in results:
                    self._result_dict[result.id] = result
                    # TODO do not add tasks with result.id < ResultMapper._expected_id

        return True
        