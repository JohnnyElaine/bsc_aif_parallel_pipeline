import numpy as np
import zmq
import pickle
import time

from multiprocessing import Process


class Collector(Process):
    def __init__(self, port: int, num_expected_messages: int):
        """
        Initialize the coordinator with the video path and edge node information.
        :param port: port the producer listens on.
        :param video_path: Path to the input video file.
        """
        super().__init__()
        self.port = port
        self.num_expected_messages = num_expected_messages

    def run(self):
        context = zmq.Context.instance()
        receiver = context.socket(zmq.PULL)
        receiver.bind(f'tcp://*:{self.port}')
        
        for i in range(self.num_expected_messages):
            msg = receiver.recv()
            s = pickle.loads(msg)
            
            print(f'Received msg: {s}')
        
        
class Worker(Process):
    def __init__(self, identifier: id, port: int, num_tasks: int, sleep_time: float):
        """
        Initialize the coordinator with the video path and edge node information.
        :param port: port the producer listens on.
        :param video_path: Path to the input video file.
        """
        super().__init__()
        self.identifier = identifier
        self.port = port
        self.num_tasks = num_tasks
        self.sleep_time = sleep_time

    def run(self):
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUSH)
        socket.connect(f"tcp://localhost:{self.port}")
    
        for i in range(self.num_tasks):

            # Do the work
            time.sleep(self.sleep_time)
            
            s = f'worker-{self.identifier} msg-{i}'
            msg = pickle.dumps(s)

            # Send results to sink
            socket.send(msg)
            
            
def main():
    port = 3000
    tasks_per_worker = 20
    num_workers = 2
    
    w1 = Worker(1, port, tasks_per_worker, 0.016)
    w2 = Worker(2, port, tasks_per_worker, 0.2)
    w3 = Worker(3, port, tasks_per_worker, 0.2)
    
    c = Collector(port, 3 * tasks_per_worker)
    
    c.start()
    
    w1.start()
    w2.start()
    w3.start()
    
    w1.join()
    w2.join()
    w3.join()
    
    c.join()
    

if __name__ == '__main__':
    main()
        
        

