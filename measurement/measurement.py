import pandas as pd

from measurement.simulation.basic_simulation import BasicSimulation
from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables

class Measurement:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.LAZY
    WORK_LOAD = WorkLoad.HIGH
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'

    @staticmethod
    def run_all_simulations():
        slo_statistics, worker_statistics = Measurement.basic_simulation(AgentType.ACTIVE_INFERENCE)

    @staticmethod
    def basic_simulation(agent_type: AgentType) -> tuple[pd.DataFrame, pd.DataFrame]:
        # dictates number of workers
        worker_processing_delays = [0, 0, 0]


        sim = BasicSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                              Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                              Measurement.WORK_LOAD, agent_type, worker_processing_delays,
                              Measurement.VID_PATH)

        return sim.run()

