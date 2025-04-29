import pandas as pd

from measurement.plotting.slo_stats_plot import plot_all_slo_stats
from measurement.plotting.worker_stats_plot import plot_all_worker_stats
from measurement.simulation.basic_simulation import BasicSimulation
from measurement.simulation.outage_and_recovery_simulation import OutageAndRecoverySimulation
from measurement.simulation.simulation_type import SimulationType
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
        #Measurement.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.BASIC)
        Measurement.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE_EXPERIMENTAL, SimulationType.BASIC)

    @staticmethod
    def run_and_plot_simulation(agent_type: AgentType, sim_type: SimulationType):
        stats = None
        match sim_type:
            case SimulationType.BASIC:
                stats = Measurement.run_basic_simulation(agent_type)
            case SimulationType.OUTAGE_AND_RECOVERY:
                stats = Measurement.run_outage_and_recovery_simulation(agent_type)
            case _:
                raise ValueError('Unknown SimulationType')

        plot_all_slo_stats(stats['slo_stats'])
        #plot_all_worker_stats(stats['worker_stats'])

    @staticmethod
    def run_basic_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        num_workers = 3

        worker_capacities = [0.3 for _ in range(num_workers)]
        #worker_capacities = [1, 1, 1]

        sim = BasicSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                              Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                              Measurement.WORK_LOAD, agent_type, Measurement.VID_PATH, worker_capacities)

        stats = sim.run()

        # add worker capacity information to df
        worker_stats = stats['worker_stats']
        worker_stats['capacity'] = [worker_capacities[identity] for identity in worker_stats.index]

        return stats

    @staticmethod
    def run_outage_and_recovery_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        outage_at = 0.25
        recovery_at = 0.75

        num_regular_workers = 2
        num_outage_workers = 1

        regular_worker_capacities = [1 for _ in range(num_regular_workers)]
        outage_worker_capacities = [1 for _ in range(num_outage_workers)]

        sim = OutageAndRecoverySimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                                          Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                                          Measurement.WORK_LOAD, agent_type, Measurement.VID_PATH,
                                          regular_worker_capacities, outage_worker_capacities,outage_at, recovery_at)

        return sim.run()
