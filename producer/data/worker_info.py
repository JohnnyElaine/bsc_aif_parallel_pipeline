from packages.message_types import RepType


class WorkerInfo:
    def __init__(self):
        self.preferred_num_of_tasks = 1
        self.instruction_backlog = []

    def has_pending_instructions(self):
        return len(self.instruction_backlog) > 0

    def get_all_pending_instructions(self):
        instructions_dict = {instruction.type: instruction.value for instruction in self.instruction_backlog}
        self.instruction_backlog.clear()
        instructions_dict['type'] = RepType.INSTRUCTION

        return instructions_dict
