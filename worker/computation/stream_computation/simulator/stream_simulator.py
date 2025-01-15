from abc import ABC

from worker.computation.stream_computation.stream_computer import StreamComputer


class StreamSimulator(StreamComputer, ABC):
    pass