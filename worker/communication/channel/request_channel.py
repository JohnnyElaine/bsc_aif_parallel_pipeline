import numpy as np
import msgpack
import zmq

from packages.data import Task
from packages.enums import WorkType, WorkLoad
from packages.message_types import ReqType
from worker.data.work_config import WorkConfig


class RequestChannel:
    def __init__(self, ip: str, port: int, identity: int):
        self._ip = ip
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt_string(zmq.IDENTITY, f'worker-{identity}')
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
        self._context.destroy()
        
    def send(self, msg):
        self._socket.send(msgpack.packb(msg))
        
    def register(self):
        req = {
            'type': ReqType.REGISTER,
        }
        self.send(req)
        msg = self._socket.recv()
        
        if msg == b'END':
            return None
        
        info = msgpack.unpackb(msg)

        return WorkConfig(WorkType.str_to_enum(info['work_type']), WorkLoad.str_to_enum(info['work_load']))

    def get_work(self) -> tuple[dict, list[Task]] | None:
        req = {
            'type': ReqType.GET_WORK,
        }

        self.send(req)
        msg = self._socket.recv_multipart()

        if msg[0] == b"END":
            return None

        info = msgpack.unpackb(msg[0])
        tasks_raw = msg[1:]

        tasks = RequestChannel.reconstruct_all_tasks(tasks_raw)

        return info, tasks

    @staticmethod
    def reconstruct_all_tasks(tasks_raw):
        return [RequestChannel.reconstruct_task(msgpack.unpackb(tasks_raw[i]), tasks_raw[i + 1]) for i in range(0, len(tasks_raw), 2)]

    @staticmethod
    def reconstruct_task(md: dict, task_buffered):
        task = np.frombuffer(task_buffered, dtype=md['dtype'])
        task = task.reshape(md['shape'])
        return Task(md['id'], task)


    def __str__(self):
        return f'({self._ip}:{self._port})'

    def __repr__(self):
        return f'({self._ip}:{self._port})'