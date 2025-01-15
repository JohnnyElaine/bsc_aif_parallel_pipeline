from importlib.metadata import metadata

import numpy as np
import zmq
import msgpack

from producer.data.task import Task


class RouterChannel:
    def __init__(self, port: int):
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)


    def bind(self):
        self._socket.bind(f'tcp://*:{self._port}')

    def close(self):
        self._socket.unbind(f'tcp://*:{self._port}')
        self._socket.close()
        self._context.destroy()

    def stop_workers(self, num_workers: int):
        for _ in range(num_workers):
            address, empty, req = self._socket.recv_multipart()
            self._socket.send_multipart([
                address,
                b'',
                b'END',
            ])

    def get_request(self) -> tuple[bytes, dict]:
        """
        Waits for a request and returns:
            address: identifier of the node that send the request (bytes)
            request: the request, containing the type (dict)
        :return: address, empty, request
        """
        address, empty, request = self._socket.recv_multipart()
        request = msgpack.unpackb(request)

        return address, request

    def send_work(self, address: bytes, tasks: list[Task]) -> None:
        msg = [address, b'']

        for task in tasks:
            metadata = dict(id=task.id, shape=task.task.shape, dtype=str(task.task.dtype))
            msg.append(msgpack.packb(metadata))
            msg.append(task)

        self._socket.send_multipart(msg)

