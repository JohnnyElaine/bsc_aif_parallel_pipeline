from abc import ABC

from aif_edge_worker.stream_computation.stream_computer import StreamComputer


class StreamSimulator(StreamComputer, ABC):
    pass