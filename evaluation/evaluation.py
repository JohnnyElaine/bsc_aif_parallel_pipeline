import pandas as pd

from evaluation.plotting.slo_stats_plot import plot_all_slo_stats
from evaluation.simulation.base_case_simulation import BaseCaseSimulation
from evaluation.simulation.simulation_type import SimulationType
from evaluation.simulation.variable_computational_budget_simulation import VariableComputationalBudgetSimulation
from evaluation.simulation.variable_computational_demand_simulation import VariableComputationalDemandSimulation
from packages.enums import WorkType, InferenceQuality
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables


class Measurement:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.EAGER
    INITIAL_INFERENCE_QUALITY = InferenceQuality.MEDIUM
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_20seconds.mp4'

    @staticmethod
    def run_all_simulations():
        #Measurement.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.BASIC)
        Measurement.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.BASIC)
        # Uncomment to run variable computational demand simulation:
        # Measurement.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND)

    @staticmethod
    def run_and_plot_simulation(agent_type: AgentType, sim_type: SimulationType):
        stats = None
        match sim_type:
            case SimulationType.BASIC:
                stats = Measurement.run_basic_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_BUDGET:
                stats = Measurement.run_variable_computational_budget_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_DEMAND:
                stats = Measurement.run_variable_computational_demand_simulation(agent_type)
            case _:
                raise ValueError('Unknown SimulationType')

        slo_stats_df = stats['slo_stats']
        plot_all_slo_stats(slo_stats_df)
        worker_stats_df = stats['worker_stats']
        #plot_all_worker_stats(worker_stats_df)

    @staticmethod
    def run_basic_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        num_workers = 3

        worker_capacities = [0.7 for _ in range(num_workers)]
        #worker_capacities = [1, 1, 1]

        sim = BaseCaseSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                                 Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                                 Measurement.INITIAL_INFERENCE_QUALITY, agent_type, Measurement.VID_PATH, worker_capacities)

        return sim.run()

    @staticmethod
    def run_variable_computational_budget_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        outage_at = 0.25
        recovery_at = 0.75

        num_regular_workers = 2
        num_outage_workers = 1

        regular_worker_capacities = [1 for _ in range(num_regular_workers)]
        outage_worker_capacities = [1 for _ in range(num_outage_workers)]

        sim = VariableComputationalBudgetSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT,
                                                    Measurement.LOCALHOST, Measurement.COLLECTOR_PORT,
                                                    WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
                                                    Measurement.INITIAL_INFERENCE_QUALITY, agent_type,
                                                    Measurement.VID_PATH, regular_worker_capacities,
                                                    outage_worker_capacities, outage_at, recovery_at)

        return sim.run()

    @staticmethod
    def run_variable_computational_demand_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        """
        Run simulation with variable computational demand using stream multiplier.
        Timeline: 0-25% single stream, 25-75% double stream, 75-100% single stream
        """
        num_workers = 3
        worker_capacities = [0.7 for _ in range(num_workers)]

        sim = VariableComputationalDemandSimulation(
            Measurement.LOCALHOST, Measurement.PRODUCER_PORT, 
            Measurement.LOCALHOST, Measurement.COLLECTOR_PORT, 
            WorkType.YOLO_DETECTION, Measurement.LOADING_MODE,
            Measurement.INITIAL_INFERENCE_QUALITY, agent_type, 
            Measurement.VID_PATH, worker_capacities,
            0.25,
            0.75,
            2
        )

        stats = sim.run()

        # Add worker capacity information to df
        worker_stats = stats['worker_stats']
        worker_stats['capacity'] = [worker_capacities[identity] for identity in worker_stats.index]

        return stats
