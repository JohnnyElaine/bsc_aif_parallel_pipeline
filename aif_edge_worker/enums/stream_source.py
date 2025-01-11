from enum import Enum


class StreamSource(Enum):
    LOCAL_MESSAGE = 'local_message'
    ZEROMQ_RADIO_DISH = 'zeromq_radio_dish'
    ZEROMQ_LOAD_BALANCING = 'zeromq_load_balancing'