from producer.elasticity.slo.slo_status import SloStatus

class SloUtil:

    CRITICAL_THRESHOLD = 1  # below this threshold the SLO is satisfied
    WARNING_THRESHOLD = 0.85

    @staticmethod
    def get_slo_status(value: float):
        if value <= SloUtil.WARNING_THRESHOLD:  # Safe zone
            return SloStatus.OK
        elif value <= SloUtil.CRITICAL_THRESHOLD:  # warning zone
            return SloStatus.WARNING
        else:
            return SloStatus.CRITICAL  # critical zone (slo not satisfied)

    @staticmethod
    def get_slo_state_probabilities(value: float):
        if value <= SloUtil.WARNING_THRESHOLD:
            # In satisfied region (OK zone)
            # As value approaches WARNING_THRESHOLD, satisfaction probability decreases
            p_ok = 1.0 - (value / SloUtil.WARNING_THRESHOLD) * 0.2  # 0.8-1.0 range
            p_warning = 1.0 - p_ok
            p_critical = 0.0
        elif value <= SloUtil.CRITICAL_THRESHOLD:
            # In warning region
            # As value approaches CRITICAL_THRESHOLD, warning probability decreases
            range_size = SloUtil.CRITICAL_THRESHOLD - SloUtil.WARNING_THRESHOLD
            position = value - SloUtil.WARNING_THRESHOLD
            p_warning = 1.0 - (position / range_size) * 0.2  # 0.8-1.0 range
            p_critical = 1.0 - p_warning
            p_ok = 0.0
        else:
            # In unsatisfied region (CRITICAL zone)
            p_critical = 1.0
            p_ok = 0.0
            p_warning = 0.0
        return [p_ok, p_warning, p_critical]