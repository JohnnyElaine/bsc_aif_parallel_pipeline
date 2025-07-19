import os
from typing import Union
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType
from evaluation.enums.directory_type import DirectoryType


class EvaluationUtils:
    """
    Utility class for consistent filepath creation across the evaluation module
    """
    
    @classmethod
    def get_filepath(cls, dir_type: DirectoryType, sim_type: Union[SimulationType, str], 
                     agent_type: Union[AgentType, str], suffix: str, 
                     file_extension: str = "parquet") -> str:
        """
        Build a complete filepath using directory type, simulation type, agent type, and suffix
        
        Args:
            dir_type: Directory type enum (IMG, SIM_DATA, METRICS, OUTPUT)
            sim_type: Simulation type enum or string
            agent_type: Agent type enum or string
            suffix: Suffix for the filename (e.g., "metrics", "slo_values", "worker_stats")
            file_extension: File extension (defaults to "parquet")
            
        Returns:
            Full filepath string
        """
        # Convert enums to lowercase strings if needed
        sim_type_name = sim_type.name.lower() if isinstance(sim_type, SimulationType) else str(sim_type).lower()
        agent_type_name = agent_type.name.lower() if isinstance(agent_type, AgentType) else str(agent_type).lower()
        
        # Build the filepath
        base_dir = dir_type.value
        sim_dir = os.path.join(base_dir, sim_type_name)
        filename = f"{agent_type_name}_{suffix}.{file_extension}"
        
        return os.path.join(sim_dir, filename)
    
    @classmethod
    def get_consolidated_filepath(cls, dir_type: DirectoryType, filename: str, 
                                 file_extension: str = "parquet") -> str:
        """
        Build filepath for consolidated files (not agent/sim-specific)
        
        Args:
            dir_type: Directory type enum
            filename: Base filename without extension
            file_extension: File extension
            
        Returns:
            Full filepath string
        """
        return os.path.join(dir_type.value, f"{filename}.{file_extension}")
    
    @classmethod
    def ensure_directory_exists(cls, filepath: str) -> None:
        """
        Ensure the directory for the given filepath exists
        
        Args:
            filepath: Full filepath (directory will be created if it doesn't exist)
        """
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
