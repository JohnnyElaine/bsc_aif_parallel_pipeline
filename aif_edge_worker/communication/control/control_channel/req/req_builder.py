from aif_edge_worker.communication.control.control_channel.constants import req_type


def register(worker_id: int):
    return {
        'req_type': req_type.REGISTER,
        'worker_id': worker_id
    }

def offload(worker_id: int, offload_frames: int, total_frames: int):
    return {
        'req_type': req_type.REGISTER,
        'worker_id': worker_id,
        'offload_frames': offload_frames,
        'total_frames': total_frames
    }

def decrease_fps(worker_id: int, decrement: int):
    return {
        'req_type': req_type.DECREASE_FPS,
        'worker_id': worker_id,
        'decrement': decrement,
    }

def increase_fps(worker_id: int, increment: int):
    return {
        'req_type': req_type.DECREASE_FPS,
        'worker_id': worker_id,
        'increment': increment,
    }

