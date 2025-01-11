import zmq

from controller.communication.control.control_channel.rep.rep_channel import RepChannel


class ControlChannel:
    def __init__(self, port: int):
        self.context = zmq.Context()
        self._rep_channel = RepChannel(port, self.context)

    def initialize(self):
        self._rep_channel.bind()

    def get_req(self):
        return self._rep_channel.recv()

    def send_rep(self, msg):
        self._rep_channel.send(msg)

    def broadcast(self):
        pass

    def multicast(self):
        pass

    def close(self):
        """
        Close all sockets associated with this context and then terminate the context.

        Warning:
            zmq.context.destroy() involves calling Socket.close(), which is NOT threadsafe.
            If there are active sockets in other threads, this must not be called.
        :return:
        """
        # Close sockets before destroying context
        self._rep_channel.close()
        self.context.destroy()

