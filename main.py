from packages.enums import WorkType, WorkLoad
from worker.enums.loading_mode import LoadingMode
from worker.global_variables import GlobalVariables
from worker.worker_config import WorkerConfig
from worker.worker import Worker
from producer.producer_config import ProducerConfig
from producer.producer import Producer


def create_workers(num: int, ip: str,port: int):
    workers = []
    for i in range(num):
        config = WorkerConfig(i, LoadingMode.LAZY, ip, port, False)
        workers.append(Worker(config))

    return workers

def main():
    port = 10000
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    num_workers = 2

    workers = create_workers(num_workers, 'localhost', port)

    for worker in workers:
        worker.start()

    producer_config = ProducerConfig(port, vid_path, WorkType.YOLO_DETECTION, WorkLoad.LOW)
    producer = Producer(producer_config)
    producer.run()

if __name__ == "__main__":
    main()
