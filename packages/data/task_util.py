import numpy as np
import msgpack

from packages.data.local_messages.task import Task

class TaskUtil:
    @staticmethod
    def reconstruct_all_tasks(tasks_raw):
        return [TaskUtil.reconstruct_task(msgpack.unpackb(tasks_raw[i]), tasks_raw[i + 1]) for i in range(0, len(tasks_raw), 2)]

    @staticmethod
    def reconstruct_task(md: dict, task_buffered):
        task = np.frombuffer(task_buffered, dtype=md['dtype'])
        task = task.reshape(md['shape'])
        return Task(md['type'], md['id'], md['stream_key'], task)