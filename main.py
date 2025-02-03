from packages.enums import ComputeType, ComputeLoad
from worker.enums.loading_mode import LoadingMode
from worker.global_variables import GlobalVariables
from worker.worker_config import WorkerConfig
from worker.worker import Worker
from producer.producer_config import ProducerConfig
from producer.producer import Producer


def create_workers(num: int, port: int):
    workers = []
    for i in range(num):
        config = WorkerConfig(i, LoadingMode.LAZY, port, False)
        workers.append(Worker(config))

    return workers

def main():
    port = 10000
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    num_nodes = 1

    workers = create_workers(num_nodes, port)

    for worker in workers:
        worker.start()

    producer_config = ProducerConfig(port, vid_path, ComputeType.YOLO_DETECTION, ComputeLoad.LOW)
    producer = Producer(producer_config)
    producer.run()

if __name__ == "__main__":
    main()
