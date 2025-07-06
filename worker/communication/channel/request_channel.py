import msgpack
import zmq

from packages.data import Task, TaskUtil
from packages.enums import WorkType, InferenceQuality, LoadingMode
from packages.network_messages import ReqType, RepType
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
        self._context.term()
        
    def send(self, msg):
        self._socket.send(msgpack.packb(msg))
        
    def register(self):
        req = dict(type=ReqType.REGISTER)

        self.send(req)
        msg = self._socket.recv()

        info = msgpack.unpackb(msg)

        if info['type'] != RepType.REGISTRATION_CONFIRMATION:
            return None

        return WorkConfig(WorkType.str_to_enum(info['work_type']), InferenceQuality.int_to_enum(info['work_load']), LoadingMode.int_to_enum(info['loading_mode']))

    def get_work(self, previous_processing_time=0.0) -> tuple[dict, list[Task]]:
        req = dict(type=ReqType.GET_WORK, previous_processing_time=previous_processing_time)

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