from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.producer_config import ProducerConfig
from producer.producer import Producer
from worker.global_variables import GlobalVariables
from worker.worker import Worker
from worker.worker_config import WorkerConfig
from collector.collector import Collector
from collector.collector_config import CollectorConfig


def create_workers(num: int, producer_ip: str, producer_port: int, collector_ip: str, collector_port: int):
    return [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port)) for i in range(num)]

def main():
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    num_workers = 2

    producer_config = ProducerConfig(10000, vid_path, WorkType.YOLO_DETECTION, WorkLoad.LOW, LoadingMode.LAZY)
    collector_config = CollectorConfig(10001)

    producer = Producer(producer_config)
    workers = create_workers(num_workers, 'localhost', producer_config.port, 'localhost', collector_config.port)
    collector = Collector(collector_config)

    for worker in workers:
        worker.start()

    collector.start()
    producer.run()

if __name__ == "__main__":
    main()
