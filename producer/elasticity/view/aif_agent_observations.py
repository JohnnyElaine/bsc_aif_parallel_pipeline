from producer.elasticity.slo.slo_manager import SloManager


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
            list[int]: [resolution_index, fps_index, inference_quality_index, queue_slo_value, memory_slo_value, global_processing_slo_value, worker_processing_slo_value]
        """
        queue_slo_status, memory_slo_status, global_processing_slo_status, worker_processing_slo_status = self._slo_manager.get_all_slo_status()
        print(self._elasticity_observations.get_current_inference_quality_state().current_index)
        return [
            self._elasticity_observations.get_current_resolution_state().current_index,
            self._elasticity_observations.get_current_fps_state().current_index,
            self._elasticity_observations.get_current_inference_quality_state().current_index,
            queue_slo_status.value,
            memory_slo_status.value,
            global_processing_slo_status.value,
            worker_processing_slo_status.value
        ]

    def get_states_indices(self):
        return self._elasticity_observations.get_current_resolution_state().current_index, self._elasticity_observations.get_current_fps_state().current_index, self._elasticity_observations.get_current_inference_quality_state().current_index,

    def get_all_slo_probabilities(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Gets all SLO probability distributions for use in the active inference agent's A matrix.
        
        Returns:
            tuple: (queue_probs, memory_probs, global_processing_probs, worker_processing_probs)
                   Each element is a list [P(OK), P(WARNING), P(CRITICAL)]
        """
        return self._slo_manager.get_all_slo_probabilities()
