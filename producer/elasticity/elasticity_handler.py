from packages.data import Instruction, InstructionType
from packages.enums import WorkLoad


class ElasticityHandler:
    def __init__(self, task_generator, request_handler):
        self._task_generator = task_generator
        self._request_handler = request_handler
        # TODO check if functions needs synchronization

    def change_work_load(self, work_load: WorkLoad):
        self._request_handler.broadcast_instruction(Instruction(InstructionType.CHANGE_WORK_LOAD, work_load.value))

    def change_fps(self, fps: int):
        self._task_generator.set_fps(fps)

    def change_resolution(self, width: int, height: int):
        self._task_generator.set_resolution(width, height)