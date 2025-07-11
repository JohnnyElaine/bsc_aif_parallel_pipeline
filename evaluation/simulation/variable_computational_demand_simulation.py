from evaluation.simulation.simulation import Simulation
from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.enums.agent_type import AgentType


class VariableComputationalDemandSimulation(Simulation):
    def __init__(self,
                 producer_ip: str,
                 producer_port: int,
                 collector_ip: str,
                 collector_port: int,
                 work_type: WorkType,
                 loading_mode: LoadingMode,
                 max_work_load: InferenceQuality,
                 agent_type: AgentType,
                 vid_path: str):
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_work_load, agent_type, vid_path)
        # TODO: continue simulation: Variable number of input streams