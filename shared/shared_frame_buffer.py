from queue import Queue

class SharedFrameBuffer:
    def __init__(self, num_nodes):
        self._read_only_list = [Queue()] * num_nodes

    def add(self, frame, index):
        self._read_only_list[index].put(frame)

    def get(self, index):
        return self._read_only_list[index].get()