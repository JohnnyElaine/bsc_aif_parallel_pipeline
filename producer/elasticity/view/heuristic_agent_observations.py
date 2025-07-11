from producer.elasticity.slo.slo_manager import SloManager
from producer.elasticity.slo.slo_status import SloStatus


class HeuristicAgentObservations:
    """
    A specialized observation view for the Heuristic Agent that combines
    elasticity parameter observations with SLO status information.
    
    This view provides a clean interface for the HeuristicAgent to observe
    both the current elasticity parameters and the system's SLO status without
    having direct access to the underlying handlers and managers.
    """

    def __init__(self, elasticity_observations_view, slo_manager: SloManager):
        """
        Initialize the heuristic agent observations view.
        
        Args:
            elasticity_observations_view: ElasticityObservationsView instance for parameter observations
            slo_manager: SloManager instance for SLO status observations
        """
        self._elasticity_observations = elasticity_observations_view
        self._slo_manager = slo_manager

    def get_all_slo_values(self) -> tuple[float, float, float, float]:
        """
        Gets all SLO values.

        Returns:
            tuple[float, float, float, float]: (queue_slo_value, memory_slo_value, global_processing_slo_value, worker_processing_slo_value)
        """
        return self._slo_manager.get_all_slo_values()

    def get_all_slo_status(self) -> tuple[SloStatus, SloStatus, SloStatus, SloStatus]:
        """
        Gets all SLO statuses.
        
        Returns:
            tuple[SloStatus, SloStatus, SloStatus, SloStatus]: (queue_status, memory_status, global_processing_status, worker_processing_status)
        """
        return self._slo_manager.get_all_slo_status()

    def get_resolution_capacity(self) -> float:
        """Gets the current resolution capacity."""
        return self._elasticity_observations.get_current_resolution_state().capacity()

    def get_fps_capacity(self) -> float:
        """Gets the current FPS capacity."""
        return self._elasticity_observations.get_current_fps_state().capacity()

    def get_inference_quality_capacity(self) -> float:
        """Gets the current inference quality capacity."""
        return self._elasticity_observations.get_current_inference_quality_state().capacity()

    def can_decrease_resolution(self) -> bool:
        """Checks if resolution can be decreased."""
        return self._elasticity_observations.get_current_resolution_state().can_decrease()

    def can_decrease_fps(self) -> bool:
        """Checks if FPS can be decreased."""
        return self._elasticity_observations.get_current_fps_state().can_decrease()

    def can_decrease_inference_quality(self) -> bool:
        """Checks if inference quality can be decreased."""
        return self._elasticity_observations.get_current_inference_quality_state().can_decrease()

    def can_increase_resolution(self) -> bool:
        """Checks if resolution can be increased."""
        return self._elasticity_observations.get_current_resolution_state().can_increase()

    def can_increase_fps(self) -> bool:
        """Checks if FPS can be increased."""
        return self._elasticity_observations.get_current_fps_state().can_increase()

    def can_increase_inference_quality(self) -> bool:
        """Checks if inference quality can be increased."""
        return self._elasticity_observations.get_current_inference_quality_state().can_increase()
