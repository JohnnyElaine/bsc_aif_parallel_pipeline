from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from worker.global_variables import GlobalVariables
from worker.worker_config import WorkerConfig
from worker.worker import Worker
from producer.producer_config import ProducerConfig
from producer.producer import Producer


def create_workers(num: int, producer_ip: str, producer_port: int):
    return [Worker(WorkerConfig(i, producer_ip, producer_port)) for i in range(num)]

def main():
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    port = 10000
    num_workers = 2

    workers = create_workers(num_workers, 'localhost', port)
    producer = Producer(ProducerConfig(port, vid_path, WorkType.YOLO_DETECTION, WorkLoad.MEDIUM, LoadingMode.LAZY))

    for worker in workers:
        worker.start()

    producer.run()

if __name__ == "__main__":
    main()
