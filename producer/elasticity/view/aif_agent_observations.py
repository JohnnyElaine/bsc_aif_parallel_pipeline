from producer.elasticity.slo.slo_manager import SloManager
from producer.elasticity.slo.slo_status import SloStatus


class AIFAgentObservations:
    """
    A specialized observation view for the Active Inference Agent that combines
    elasticity parameter observations with SLO status information.
    
    This view provides a clean interface for the ActiveInferenceAgent to observe
    both the current elasticity parameters and the system's SLO status without
    having direct access to the underlying handlers and managers.
    """

    def __init__(self, elasticity_observations_view, slo_manager: SloManager):
        """
        Initialize the AIF agent observations view.
        
        Args:
            elasticity_observations_view: ElasticityObservationsView instance for parameter observations
            slo_manager: SloManager instance for SLO status observations
        """
        self._elasticity_observations = elasticity_observations_view
        self._slo_manager = slo_manager

    def get_observations(self) -> list[int]:
        """
        Gets all observations that the ActiveInferenceAgent needs as a list of indices.
        
        This method provides the exact format expected by the AIF agent for its observation vector.

        Returns:
            list[int]: [resolution_index, fps_index, inference_quality_index, queue_slo_value, memory_slo_value]
        """
        return [
            self._elasticity_observations.get_current_resolution_state().current_index,
            self._elasticity_observations.get_current_fps_state().current_index,
            self._elasticity_observations.get_current_inference_quality_state().current_index,
            self._slo_manager.get_queue_size_slo_status().value,
            self._slo_manager.get_memory_slo_status().value
        ]
