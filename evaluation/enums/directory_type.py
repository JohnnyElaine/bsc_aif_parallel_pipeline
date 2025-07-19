from enum import Enum


class DirectoryType(Enum):
    """Enum for different output directory types in the evaluation system"""
    
    IMG = "out/img"
    SIM_DATA = "out/sim-data"  
    METRICS = "out/metrics"
    OUTPUT = "out"
