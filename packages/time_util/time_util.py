import time

def enforce_target_fps(iteration_start_time: float, target_frame_interval: float):
    iteration_duration = time.perf_counter() - iteration_start_time
    wait_time = max(target_frame_interval - iteration_duration, 0)
    if wait_time > 0:
        time.sleep(wait_time)