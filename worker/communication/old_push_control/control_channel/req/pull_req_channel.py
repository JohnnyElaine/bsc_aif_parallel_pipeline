import zmq
import msgpack


class PullReqChannel:
    def __init__(self, ip: str, port: int, zmq_context: zmq.Context):
        self._ip = ip
        self._port = port
        self._socket = zmq_context.socket(zmq.REQ)
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
        self._socket.close()

    def send_req_get_rep(self, msg):
        self._socket.send(msgpack.packb(msg))
        return msgpack.unpackb(self._socket.get_req())