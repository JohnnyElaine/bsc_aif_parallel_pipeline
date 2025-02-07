import msgpack
import zmq
from packages.data import Task

class PushChannel:
    def __init__(self, ip: str, port: int, identity: int, zmq_context):
        self._ip = ip
        self._port = port
        self._context = zmq_context
        self._socket = self._context.socket(zmq.PUSH)
        self._is_connected = False

    def connect(self):
        self._socket.connect(f'tcp://{self._ip}:{self._port}')
        self._is_connected = True

    def disconnect(self):
        self._socket.disconnect(f'tcp://{self._ip}:{self._port}')
        self._is_connected = False

    def close(self):
        if self._is_connected:
            self.disconnect()

        # Close sockets before destroying context
        self._socket.close()
        self._context.destroy() # TODO check what happens if context is closed somewhere else first (i.e. in RequestChannel^)
        
    def send_results(self, results: list[Task]):
        msg = list()

        for result in results:
            metadata = dict(id=result.id, shape=result.task.shape, dtype=str(result.task.dtype))
            msg.append(msgpack.packb(metadata))# send metadata first
            msg.append(result.task) # send raw numpy array after, use the implemented buffer interface

        self._socket.send_multipart(msg)

    def __str__(self):
        return f'({self._ip}:{self._port})'

    def __repr__(self):
        return f'({self._ip}:{self._port})'