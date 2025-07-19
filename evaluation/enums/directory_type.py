import os
from enum import Enum

OUTPUT_DIR = "out"

class DirectoryType(Enum):
    """Enum for different output directory types in the evaluation system"""
    
    IMG = os.path.join(OUTPUT_DIR, "img")
    SIM_DATA = os.path.join(OUTPUT_DIR, "sim-data")
    METRICS = os.path.join(OUTPUT_DIR, "metrics")
    OUTPUT = "out"
