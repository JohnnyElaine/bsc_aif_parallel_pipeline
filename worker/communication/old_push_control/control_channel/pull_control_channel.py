import zmq

from worker.communication.old_push_control.control_channel.req.pull_req_channel import PullReqChannel


class PullControlChannel:
    def __init__(self, ip: str, port: int):
        self.context = zmq.Context()
        self._req_channel = PullReqChannel(ip, port, self.context)

    def send_req(self, req):
        self._req_channel.send_req_get_rep(req)

    def connect(self):
        self._req_channel.connect()

    def close(self):
        """
        Close all sockets associated with this context and then terminate the context.

        Warning:
            zmq.context.destroy() involves calling Socket.close(), which is NOT threadsafe.
            If there are active sockets in other threads, this must not be called.
        :return:
        """
        # Close sockets before destroying context
        self._req_channel.close()
        self.context.destroy()

