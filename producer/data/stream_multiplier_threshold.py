from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class StreamMultiplierThreshold:
    """
    Represents a single entry in the stream multiplier schedule.
    
    Attributes:
        frame: The frame when the multiplier is applied
        multiplier: The stream multiplier value (1 = single stream, 2 = double, etc.)
    """
    frame: int
    multiplier: int = field(compare=False)
