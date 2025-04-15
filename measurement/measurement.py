import pandas as pd

from measurement.plotting.slo_stats_plot import plot_all
from measurement.simulation.basic_simulation import BasicSimulation
from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables

class Measurement:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.EAGER
    WORK_LOAD = WorkLoad.MEDIUM
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_20seconds.mp4'

    @staticmethod
    def run_all_simulations():
        stats = Measurement.basic_simulation(AgentType.ACTIVE_INFERENCE)
        slo_stats = stats['slo_stats']
        worker_stats = stats['worker_stats']

        print(slo_stats.info())
        print(slo_stats)
        print(worker_stats.info())
        print(worker_stats)

        plot_all(slo_stats)

    @staticmethod
    def basic_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        num_workers = 3
        #worker_capacities = [0.8 for i in range(num_workers)]
        worker_capacities = [1 for _ in range(num_workers)]

        sim = BasicSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                              Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                              Measurement.WORK_LOAD, agent_type, worker_capacities,
                              Measurement.VID_PATH)

        return sim.run()

