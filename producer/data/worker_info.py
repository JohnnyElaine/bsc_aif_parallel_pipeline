from packages.data.local_messages.task import Task
from packages.network_messages import RepType
from producer.statistics.moving_average import MovingAverage


class WorkerInfo:
    def __init__(self):
        self._preferred_num_of_tasks = 1
        self._change_backlog = []
        self._processing_time_moving_average = MovingAverage(window_size=30)
        self.has_received_end_message = False

    def add_processing_time(self, t: float):
        self._processing_time_moving_average.add(t)

    def get_avg_processing_time(self) -> float:
        return self._processing_time_moving_average.average()

    def add_change(self, change: Task):
        self._change_backlog.append(change)

    def has_pending_changes(self):
        return len(self._change_backlog) > 0

    def get_all_pending_changes(self) -> dict:
        # format changes as dict.
        # If there are multiple changes of the same type, only the most recent one (higher index in list) is used
        changes = {change.type: change.data.item() for change in self._change_backlog}
        self._change_backlog.clear()
        changes['type'] = RepType.CHANGE

        return changes
