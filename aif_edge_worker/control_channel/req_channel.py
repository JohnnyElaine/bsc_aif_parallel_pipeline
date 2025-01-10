import zmq
import msgpack


class ReqChannel:
    def __init__(self, hostname: str, port: int, zmq_context: zmq.Context):
        self._hostname = hostname
        self._port = port
        self._socket = zmq_context.socket(zmq.REQ)
        self._is_connected = False

    def connect(self):
        self._socket.connect(f'tcp://{self._hostname}:{self._port}')
        self._is_connected = True

    def disconnect(self):
        self._socket.disconnect(f'tcp://{self._hostname}:{self._port}')
        self._is_connected = False

    def close(self):
        if self._is_connected:
            self._socket.close()
        self._socket.close()

    def send(self, data):
        self._socket.send(msgpack.packb(data))

    def recv(self):
        return msgpack.unpackb(self._socket.recv())