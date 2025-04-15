import msgpack
import zmq

from packages.data import Task, TaskType
from packages.network_messages import RepType


class PushChannel:
    def __init__(self, ip: str, port: int, send_timeout_ms=2000, linger_timeout_ms=2000):
        self._ip = ip
        self._port = port
        self._context = zmq.Context().instance()
        self._socket = self._context.socket(zmq.PUSH)

        # Controls what happens to unsent messages when the socket is closed
        # If this is not set zeroMQ will block indefinitely
        self._socket.setsockopt(zmq.LINGER, linger_timeout_ms)
        # Sets the maximum time a send operation will block waiting to complete
        self._socket.setsockopt(zmq.SNDTIMEO, send_timeout_ms)

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
        self._context.term()

    def send_info(self, info: dict):
        msg = [msgpack.packb(info)]
        self._socket.send_multipart(msg)
        #self._socket.send_multipart(msg, zmq.NOBLOCK) # raises ZMQError

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