import msgpack
import zmq

from packages.data import Task, TaskUtil
from packages.enums import WorkType, WorkLoad
from packages.message_types import ReqType, RepType
from worker.data.work_config import WorkConfig


class RequestChannel:
    def __init__(self, ip: str, port: int, identity: int):
        self._ip = ip
        self._port = port
        self._context = zmq.Context().instance()
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

        info = msgpack.unpackb(msg)

        if info['type'] != RepType.REGISTRATION_CONFIRMATION:
            return None

        return WorkConfig(WorkType.str_to_enum(info['work_type']), WorkLoad.str_to_enum(info['work_load']))

    def get_work(self) -> tuple[dict, list[Task]] | None:
        req = {
            'type': ReqType.GET_WORK,
        }

        self.send(req)
        msg = self._socket.recv_multipart()

        info = msgpack.unpackb(msg[0])
        tasks_raw = msg[1:]

        tasks = TaskUtil.reconstruct_all_tasks(tasks_raw)

        return info, tasks

    def __str__(self):
        return f'({self._ip}:{self._port})'

    def __repr__(self):
        return f'({self._ip}:{self._port})'