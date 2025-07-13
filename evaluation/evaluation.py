import pandas as pd

from evaluation.plotting.slo_stats_plot import plot_all_slo_stats
from evaluation.simulation.cases.base_case_simulation import BaseCaseSimulation
from evaluation.simulation.simulation_type import SimulationType
from evaluation.simulation.cases.variable_computational_budget_simulation import VariableComputationalBudgetSimulation
from evaluation.simulation.cases.variable_computational_demand_simulation import VariableComputationalDemandSimulation
from packages.enums import WorkType, InferenceQuality
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables


class Evaluation:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.EAGER
    INITIAL_INFERENCE_QUALITY = InferenceQuality.MEDIUM
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_20seconds.mp4'

    @staticmethod
    def run_all_simulations():
        #Evaluation.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.BASIC)
        Evaluation.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND)
        #Evaluation.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET)

    @staticmethod
    def run_and_plot_simulation(agent_type: AgentType, sim_type: SimulationType):
        stats = None
        match sim_type:
            case SimulationType.BASIC:
                stats = Evaluation.run_base_case_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_BUDGET:
                stats = Evaluation.run_variable_computational_budget_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_DEMAND:
                stats = Evaluation.run_variable_computational_demand_simulation(agent_type)
            case _:
                raise ValueError('Unknown SimulationType')

        slo_stats_df = stats['slo_stats']
        plot_all_slo_stats(slo_stats_df)
        worker_stats_df = stats['worker_stats']
        #plot_all_worker_stats(worker_stats_df)

    @staticmethod
    def run_base_case_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        num_workers = 1

        worker_capacities = [1 for _ in range(num_workers)]

        sim = BaseCaseSimulation(Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT, Evaluation.LOCALHOST,
                                 Evaluation.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
                                 Evaluation.INITIAL_INFERENCE_QUALITY, agent_type, Evaluation.VID_PATH, worker_capacities)

        return sim.run()

    @staticmethod
    def run_variable_computational_budget_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        outage_at = 0.33
        recovery_at = 0.66

        num_regular_workers = 2
        num_outage_workers = 1

        regular_worker_capacities = [1 for _ in range(num_regular_workers)]
        outage_worker_capacities = [1 for _ in range(num_outage_workers)]

        sim = VariableComputationalBudgetSimulation(Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT,
                                                    Evaluation.LOCALHOST, Evaluation.COLLECTOR_PORT,
                                                    WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
                                                    Evaluation.INITIAL_INFERENCE_QUALITY, agent_type,
                                                    Evaluation.VID_PATH, regular_worker_capacities,
                                                    outage_worker_capacities, outage_at, recovery_at)

        return sim.run()

    @staticmethod
    def run_variable_computational_demand_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        """
        Run simulation with variable computational demand using stream multiplier.
        Timeline: 0-25% single stream, 25-75% double stream, 75-100% single stream
        """
        num_workers = 3
        worker_capacities = [1 for _ in range(num_workers)]

        increase_at = 0.33
        decrease_at = 0.66
        increase_to = 2
        decrease_to = 1

        sim = VariableComputationalDemandSimulation(
            Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT,
            Evaluation.LOCALHOST, Evaluation.COLLECTOR_PORT,
            WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
            Evaluation.INITIAL_INFERENCE_QUALITY, agent_type,
            Evaluation.VID_PATH, worker_capacities,
            increase_at,
            decrease_at,
            increase_to,
            decrease_to
        )

        stats = sim.run()

        # Add worker capacity information to df
        worker_stats = stats['worker_stats']
        worker_stats['capacity'] = [worker_capacities[identity] for identity in worker_stats.index]

        return stats
