from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class StreamMultiplierEntry:
    """
    Represents a single entry in the stream multiplier schedule.
    
    Attributes:
        multiplier: The stream multiplier value (1 = single stream, 2 = double, etc.)
        frame_percentage: The percentage of total frames when this multiplier takes effect (0.0 to 1.0)
    """
    frame_percentage: float
    multiplier: int = field(compare=False)
