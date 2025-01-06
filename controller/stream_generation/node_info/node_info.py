class NodeInfo:
    def __init__(self, identifier: int, ip: str, port: int, connection=None):
        self.ip = ip
        self.port = port
        self.identifier = identifier
        self.data_stream_connection = connection
        self.control_stream_connection = None
        self._offloading_preference = None

    def is_offloading(self):
        if self._offloading_preference is None:
            return False

        return self._offloading_preference.increment_and_decide()