import zmq
import msgpack


class RepChannel:
    def __init__(self, port: int, zmq_context: zmq.Context):
        self._port = port
        self._socket = zmq_context.socket(zmq.REP)

    def bind(self):
        self._socket.bind(f"tcp://*:{self._port}")

    def close(self):
        self._socket.unbind(f"tcp://*:{self._port}")
        self._socket.close()

    def send(self, data):
        self._socket.send(msgpack.packb(data))

    def recv(self):
        return msgpack.unpackb(self._socket.recv())