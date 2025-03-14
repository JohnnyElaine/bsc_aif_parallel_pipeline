import zmq

from packages.data import Task, TaskUtil


class PullChannel:
    def __init__(self, port: int):
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        
    def bind(self):
        self._socket.bind(f'tcp://*:{self._port}')

    def close(self):
        self._socket.unbind(f'tcp://*:{self._port}')
        self._socket.close()
        self._context.destroy()
        
    def get_results(self) -> list[Task]:
        # TODO: What to do when steam is done
        tasks_raw = self._socket.recv_multipart()
        
        results = TaskUtil.reconstruct_all_tasks(tasks_raw)
        
        return results
        
    def __str__(self):
        return f'(socket={self._socket}, port={self._port})'

    def __repr__(self):
        return f'(socket={self._socket}, port={self._port})'