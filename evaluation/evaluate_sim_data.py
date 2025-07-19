"""
Evaluation Script for Simulation Data Analysis

This script handles all the analysis tasks after simulation data has been saved:
1. Read the simulation data
2. Create plots for each simulation and agent using slo_stats_plot
3. Calculate values using slo_calc.py for each simulation and agent
4. For each simulation type: compare the calculated values for active inference and heuristic agent using comparison.py

Usage:
    python evaluate_sim_data.py
"""

from evaluation.plotting.slo_stats_plot import plot_all_slo_stats_from_file
from evaluation.plotting.worker_stats_plot import plot_all_worker_stats_from_file
from evaluation.calc.slo_calc import calculate_and_save_slo_metrics_from_file
from evaluation.calc.comparison import compare_agent_metrics
from producer.enums.agent_type import AgentType
from evaluation.simulation.simulation_type import SimulationType


def evaluate():
    """Run evaluation pipeline for all simulation data"""
    
    # Configuration using enums
    agent_types = [AgentType.ACTIVE_INFERENCE_RELATIVE_CONTROL, AgentType.HEURISTIC]
    sim_types = [SimulationType.BASIC, SimulationType.VARIABLE_COMPUTATIONAL_DEMAND, SimulationType.VARIABLE_COMPUTATIONAL_BUDGET]
    
    # Step 1 & 2: Read simulation data and create plots for each simulation and agent
    for sim_type in sim_types:
        for agent_type in agent_types:
            plot_all_slo_stats_from_file(agent_type, sim_type)
            plot_all_worker_stats_from_file(agent_type, sim_type)
    
    # Step 3: Calculate values using slo_calc.py for each simulation and agent
    for sim_type in sim_types:
        for agent_type in agent_types:
            calculate_and_save_slo_metrics_from_file(agent_type, sim_type)
    
    # Step 4: For each simulation type, compare the calculated values between agents
    compare_agent_metrics()


if __name__ == "__main__":
    evaluate()
