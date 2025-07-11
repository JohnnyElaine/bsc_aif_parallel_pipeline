import producer.elasticity.handler.possible_values.aspect_ratios as ASPECT_RATIOS
from packages.enums import InferenceQuality
from producer.data.resolution import Resolution
from producer.elasticity.handler.possible_values.resolutions import RESOLUTIONS_16_9, RESOLUTIONS_4_3


def generate_possible_resolutions(max_res: Resolution) -> list[Resolution]:
    """
    Generates a list of possible resolution values based on the initial configuration's aspect ratio.

    Returns:
        list[Resolution]: A list of possible resolution values.
    """
    match max_res.get_aspect_ratio():
        case ASPECT_RATIOS.ASPECT_RATIO_16_9:
            all_resolutions = RESOLUTIONS_16_9
        case ASPECT_RATIOS.ASPECT_RATIO_4_3:
            all_resolutions = RESOLUTIONS_4_3
        case _:
            all_resolutions = RESOLUTIONS_16_9

    return [res for res in all_resolutions if res.pixels <= max_res.pixels]

def generate_possible_fps(max_fps: int):
    """
    Returns possible FPS values from max_fps down to 10, stepping by -2.

    Returns:
        list[int]: A list of possible FPS values.
    """
    a = list(range(max_fps, 9, -2))
    a.reverse()
    return a

def generate_possible_work_loads(max_work_load: InferenceQuality) -> list[InferenceQuality]:
    """
    Generates a list of possible workload values.

    Returns:
        list[InferenceQuality]: A list of possible workload values.
    """
    return [w for w in list(InferenceQuality) if w.value <= max_work_load.value]