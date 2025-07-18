import pandas as pd

from evaluation.plotting.slo_stats_plot import plot_all_slo_stats
from evaluation.plotting.worker_stats_plot import plot_all_worker_stats
from evaluation.calc.slo_calc import calculate_and_save_slo_metrics
from evaluation.simulation.cases.base_case_simulation import BaseCaseSimulation
from evaluation.simulation.simulation_type import SimulationType
from evaluation.simulation.cases.variable_computational_budget_simulation import VariableComputationalBudgetSimulation
from evaluation.simulation.cases.variable_computational_demand_simulation import VariableComputationalDemandSimulation
from packages.enums import WorkType, InferenceQuality
from packages.enums.loading_mode import LoadingMode
from producer.data.stream_multiplier_entry import StreamMultiplierEntry
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables


class Evaluation:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.EAGER
    INITIAL_INFERENCE_QUALITY = InferenceQuality.HIGH
    NUM_WORKERS = 3
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_450seconds.mp4'

    @staticmethod
    def run_all_simulations():
        Evaluation.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET)
        Evaluation.run_and_plot_simulation(AgentType.HEURISTIC, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET)

        #Evaluation.run_aif_agent_simulations()
        #Evaluation.run_heuristic_agent_simulations()

    @staticmethod
    def run_aif_agent_simulations():
        eval_sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND,
                          SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

        for sim_type in eval_sim_types:
            Evaluation.run_and_plot_simulation(AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, sim_type)

    @staticmethod
    def run_heuristic_agent_simulations():
        eval_sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND,
                          SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

        for sim_type in eval_sim_types:
            Evaluation.run_and_plot_simulation(AgentType.HEURISTIC, sim_type)

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
        
        # Create descriptive names for the plots
        agent_type_name = agent_type.name.lower()
        sim_type_name = sim_type.name.lower()
        
        plot_all_slo_stats(slo_stats_df, agent_type_name, sim_type_name)
        
        # Calculate and save SLO metrics
        calculate_and_save_slo_metrics(slo_stats_df, agent_type_name, sim_type_name)
        
        # Plot worker statistics
        plot_all_worker_stats(stats['worker_stats'], agent_type_name, sim_type_name)

    @staticmethod
    def run_base_case_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        worker_capacities = [0.6, 0.5, 0.4]

        sim = BaseCaseSimulation(Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT, Evaluation.LOCALHOST,
                                 Evaluation.COLLECTOR_PORT, WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
                                 Evaluation.INITIAL_INFERENCE_QUALITY, agent_type, Evaluation.VID_PATH, worker_capacities)

        return sim.run()


    @staticmethod
    def run_variable_computational_demand_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        """
        Run simulation with variable computational demand using stream multiplier.
        """
        worker_capacities = [0.8, 0.75, 0.7]

        stream_multiplier_schedule = [
            StreamMultiplierEntry(0.25, 3),
            StreamMultiplierEntry(0.5, 2),
            StreamMultiplierEntry(0.75, 1),
        ]

        sim = VariableComputationalDemandSimulation(
            Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT,
            Evaluation.LOCALHOST, Evaluation.COLLECTOR_PORT,
            WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
            Evaluation.INITIAL_INFERENCE_QUALITY, agent_type,
            Evaluation.VID_PATH, worker_capacities,
            stream_multiplier_schedule
        )

        stats = sim.run()

        # Add worker capacity information to df
        worker_stats = stats['worker_stats']
        worker_stats['capacity'] = [worker_capacities[identity] for identity in worker_stats.index]

        return stats

    @staticmethod
    def run_variable_computational_budget_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        outage_at = 0.33
        recovery_at = 0.66

        num_outage_workers = Evaluation.NUM_WORKERS // 2
        num_regular_workers = Evaluation.NUM_WORKERS - num_outage_workers

        regular_worker_capacities = [0.5]
        outage_worker_capacities = [0.5]

        sim = VariableComputationalBudgetSimulation(Evaluation.LOCALHOST, Evaluation.PRODUCER_PORT,
                                                    Evaluation.LOCALHOST, Evaluation.COLLECTOR_PORT,
                                                    WorkType.YOLO_DETECTION, Evaluation.LOADING_MODE,
                                                    Evaluation.INITIAL_INFERENCE_QUALITY, agent_type,
                                                    Evaluation.VID_PATH, regular_worker_capacities,
                                                    outage_worker_capacities, outage_at, recovery_at)

        return sim.run()
