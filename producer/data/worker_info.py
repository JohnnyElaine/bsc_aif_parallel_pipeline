from packages.data import Change
from packages.data.local_messages.task import Task
from packages.network_messages import RepType


class WorkerInfo:
    def __init__(self):
        self.preferred_num_of_tasks = 1
        self.change_backlog = []
        self.has_received_end_message = False

    def add_change(self, change: Task):
        self.change_backlog.append(change)

    def has_pending_changes(self):
        return len(self.change_backlog) > 0

    def get_all_pending_changes(self):
        # format changes as dict.
        # If there are multiple changes of the same type, only the most recent one (higher index in list) is used
        changes = {change.type: change.data.item() for change in self.change_backlog}
        self.change_backlog.clear()
        changes['type'] = RepType.CHANGE

        return changes
