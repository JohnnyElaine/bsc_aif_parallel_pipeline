import zmq

from threading import Thread


class ControlChannel:
    def __init__(self):
        self.context = zmq.Context()

    def close(self):
        """
        Close all sockets associated with this context and then terminate the context.

        Warning:
            zmq.context.destroy() involves calling Socket.close(), which is NOT threadsafe.
            If there are active sockets in other threads, this must not be called.
        :return:
        """
        self.context.destroy()

