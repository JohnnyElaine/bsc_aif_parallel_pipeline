import msgpack
import zmq
from packages.data import Task, TaskType
from packages.network_messages import RepType


class PushChannel:
    def __init__(self, ip: str, port: int):
        self._ip = ip
        self._port = port
        self._context = zmq.Context().instance()
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
        self._context.destroy() # TODO check what happens if context is closed somewhere else first (i.e. in RequestChannel)

    def send_info(self, info: dict):
        msg = [msgpack.packb(info)]
        self._socket.send_multipart(msg)
        
    def send_results(self, results: list[Task]):
        info = {'type': RepType.WORK}
        msg = [msgpack.packb(info)]

        for result in results:
            metadata = dict(id=result.id, type=TaskType.COLLECT, shape=result.data.shape, dtype=str(result.data.dtype))
            msg.append(msgpack.packb(metadata)) # send metadata first
            msg.append(result.data) # send raw numpy array after, uses the implemented buffer interface

        self._socket.send_multipart(msg)

    def __str__(self):
        return f'({self._ip}:{self._port})'

    def __repr__(self):
        return f'({self._ip}:{self._port})'