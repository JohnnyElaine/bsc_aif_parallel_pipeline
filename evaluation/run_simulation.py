import pandas as pd
import os

from evaluation.simulation.cases.base_case_simulation import BaseCaseSimulation
from evaluation.simulation.simulation_type import SimulationType
from evaluation.simulation.cases.variable_computational_budget_simulation import VariableComputationalBudgetSimulation
from evaluation.simulation.cases.variable_computational_demand_simulation import VariableComputationalDemandSimulation
from packages.enums import WorkType, InferenceQuality
from packages.enums.loading_mode import LoadingMode
from producer.data.stream_multiplier_entry import StreamMultiplierEntry
from producer.enums.agent_type import AgentType
from worker.global_variables import WorkerGlobalVariables
from evaluation.evaluation_utils import EvaluationUtils
from evaluation.enums.directory_type import DirectoryType


class RunSimulation:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    LOADING_MODE = LoadingMode.EAGER
    INITIAL_INFERENCE_QUALITY = InferenceQuality.HIGH
    NUM_WORKERS = 3
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_450seconds.mp4'
    #VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w]_5seconds.mp4'

    @staticmethod
    def run_all_simulations():
        RunSimulation.run_aif_agent_simulations()
        #RunSimulation.run_heuristic_agent_simulations() # Do not re-reun heuristic simulations unless changes to the agent have been made

    @staticmethod
    def run_aif_agent_simulations():
        eval_sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND,
                          SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

        for sim_type in eval_sim_types:
            RunSimulation.run(AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, sim_type)

    @staticmethod
    def run_heuristic_agent_simulations():
        eval_sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND,
                          SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]

        for sim_type in eval_sim_types:
            RunSimulation.run(AgentType.HEURISTIC, sim_type)

    @staticmethod
    def run(agent_type: AgentType, sim_type: SimulationType):
        stats = None
        match sim_type:
            case SimulationType.BASIC:
                stats = RunSimulation.run_base_case_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_BUDGET:
                stats = RunSimulation.run_variable_computational_budget_simulation(agent_type)
            case SimulationType.VARIABLE_COMPUTATIONAL_DEMAND:
                stats = RunSimulation.run_variable_computational_demand_simulation(agent_type)
            case _:
                raise ValueError('Unknown SimulationType')

        RunSimulation.save_simulation_statistics(stats['slo_stats'], stats['worker_stats'], agent_type, sim_type)

    @staticmethod
    def save_simulation_statistics(slo_stats_df: pd.DataFrame, worker_stats_df: pd.DataFrame, agent_type: AgentType, sim_type: SimulationType, output_dir: str = "out/sim-data"):
        """
        Save simulation statistics to files for later analysis
        
        Args:
            slo_stats_df: DataFrame containing SLO statistics
            worker_stats_df: DataFrame containing worker statistics
            agent_type: The agent type enum
            sim_type: The simulation type enum
            output_dir: Directory to save statistics files
        """
        slo_stats_filepath = EvaluationUtils.get_filepath(DirectoryType.SIM_DATA, sim_type, agent_type, "slo_stats", "csv")
        worker_stats_filepath = EvaluationUtils.get_filepath(DirectoryType.SIM_DATA, sim_type, agent_type, "worker_stats", "csv")
        
        EvaluationUtils.ensure_directory_exists(slo_stats_filepath)
        EvaluationUtils.ensure_directory_exists(worker_stats_filepath)
        
        slo_stats_df.to_csv(slo_stats_filepath, index=True)
        worker_stats_df.to_csv(worker_stats_filepath, index=True)

    @staticmethod
    def run_base_case_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        worker_capacities = [0.6, 0.5, 0.4]

        sim = BaseCaseSimulation(RunSimulation.LOCALHOST, RunSimulation.PRODUCER_PORT, RunSimulation.LOCALHOST,
                                 RunSimulation.COLLECTOR_PORT, WorkType.YOLO_DETECTION, RunSimulation.LOADING_MODE,
                                 RunSimulation.INITIAL_INFERENCE_QUALITY, agent_type, RunSimulation.VID_PATH, worker_capacities)

        return sim.run()


    @staticmethod
    def run_variable_computational_demand_simulation(agent_type: AgentType) -> dict[str, pd.DataFrame]:
        """
        Run simulation with variable computational demand using stream multiplier.
        """
        worker_capacities = [0.8, 0.75, 0.7]

        stream_multiplier_schedule = [
            StreamMultiplierEntry(0.2, 2),
            StreamMultiplierEntry(0.40, 3),
            StreamMultiplierEntry(0.6, 2),
            StreamMultiplierEntry(0.8, 1),
        ]

        sim = VariableComputationalDemandSimulation(
            RunSimulation.LOCALHOST, RunSimulation.PRODUCER_PORT,
            RunSimulation.LOCALHOST, RunSimulation.COLLECTOR_PORT,
            WorkType.YOLO_DETECTION, RunSimulation.LOADING_MODE,
            RunSimulation.INITIAL_INFERENCE_QUALITY, agent_type,
            RunSimulation.VID_PATH, worker_capacities,
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

        num_outage_workers = RunSimulation.NUM_WORKERS // 2
        num_regular_workers = RunSimulation.NUM_WORKERS - num_outage_workers

        regular_worker_capacities = [0.5]
        outage_worker_capacities = [0.5]

        sim = VariableComputationalBudgetSimulation(RunSimulation.LOCALHOST, RunSimulation.PRODUCER_PORT,
                                                    RunSimulation.LOCALHOST, RunSimulation.COLLECTOR_PORT,
                                                    WorkType.YOLO_DETECTION, RunSimulation.LOADING_MODE,
                                                    RunSimulation.INITIAL_INFERENCE_QUALITY, agent_type,
                                                    RunSimulation.VID_PATH, regular_worker_capacities,
                                                    outage_worker_capacities, outage_at, recovery_at)

        return sim.run()
