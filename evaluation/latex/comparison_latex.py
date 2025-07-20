"""
LaTeX table generation for comparison results

This module provides functions to generate LaTeX tables from comparison DataFrames
created by the comparison module.
"""

import os
import pandas as pd
from evaluation.enums.directory_type import DirectoryType
from evaluation.evaluation_utils import EvaluationUtils


def create_latex_table_summary(comparison_df: pd.DataFrame) -> str:
    """
    Create a LaTeX table from the comparison DataFrame with the most important metrics
    
    Args:
        comparison_df: Multi-indexed DataFrame with comparison results
        
    Returns:
        LaTeX table string
    """
    # Define the most important metrics to include
    important_metrics = [
        'time_all_slo_fulfilled_simultaneously_percent',
        'average_slo_fulfillment_rate_percent', 
        'average_overall_stream_quality_score',
        'maximum_consecutive_timesteps_with_slo_violations',
        'average_timesteps_to_reach_stable_quality_configuration'
    ]
    
    # Define metric display names
    metric_names = {
        'time_all_slo_fulfilled_simultaneously_percent': 'Time All SLOs Met',
        'average_slo_fulfillment_rate_percent': 'Avg. SLO Fulfillment',
        'average_overall_stream_quality_score': 'Stream Quality Score',
        'maximum_consecutive_timesteps_with_slo_violations': 'Max SLO Violation Streak',
        'average_timesteps_to_reach_stable_quality_configuration': 'Timesteps to reach stable config coeff.'
    }
    
    # Define simulation type display names
    sim_names = {
        'basic': 'Base',
        'variable_computational_budget': 'Budget',
        'variable_computational_demand': 'Demand'
    }
    
    # Start LaTeX table
    latex = []
    latex.append(r'\begin{table}[h!]')
    latex.append(r'\centering')
    latex.append(r'\caption{Summary of Core Evaluation Metrics}')
    latex.append(r'\label{tab:results_summary}')
    latex.append(r'\begin{tabular}{@{}lllll@{}}')
    latex.append(r'\toprule')
    latex.append(r'\textbf{Scenario} & \textbf{Metric} & \textbf{AIF} & \textbf{Heuristic} & \textbf{$\Delta$} \\')
    latex.append(r'\midrule')
    
    # Process each simulation type
    sim_types = ['basic', 'variable_computational_budget', 'variable_computational_demand']
    
    for i, sim_type in enumerate(sim_types):
        sim_display_name = sim_names[sim_type]
        
        # Add multirow for first metric of each simulation
        first_metric = True
        
        for metric in important_metrics:
            try:
                # Get the data for this simulation and metric
                row = comparison_df.loc[(sim_type, metric)]
                aif_value = row['aif_agent']
                heuristic_value = row['heuristic']
                delta_value = row['delta']
                
                # Format values based on metric type
                if metric == 'maximum_consecutive_timesteps_with_slo_violations':
                    # Integer values for timesteps
                    aif_str = f"{int(aif_value)}"
                    heuristic_str = f"{int(heuristic_value)}"
                elif metric == 'average_timesteps_to_reach_stable_quality_configuration':
                    # Format to 1 decimal place for timesteps
                    aif_str = f"{aif_value:.1f}"
                    heuristic_str = f"{heuristic_value:.1f}"
                else:
                    # Format to 3 decimal places for percentages and scores
                    aif_str = f"{aif_value:.3f}"
                    heuristic_str = f"{heuristic_value:.3f}"
                
                # Format delta with proper sign and percentage
                if delta_value >= 0:
                    delta_str = f"+{delta_value:.2f}\\%"
                else:
                    delta_str = f"{delta_value:.2f}\\%"
                
                # Create the row
                if first_metric:
                    # First row of the simulation type uses multirow
                    latex.append(f"\\multirow{{5}}{{*}}{{{sim_display_name}}} & {metric_names[metric]} & {aif_str} & {heuristic_str} & {delta_str} \\\\")
                    first_metric = False
                else:
                    # Subsequent rows start with empty cell
                    latex.append(f"& {metric_names[metric]} & {aif_str} & {heuristic_str} & {delta_str} \\\\")
                
            except KeyError:
                print(f"Warning: Metric '{metric}' not found for simulation '{sim_type}'")
                continue
        
        # Add midrule after each simulation type except the last
        if i < len(sim_types) - 1:
            latex.append(r'\midrule')
    
    # Close the table
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')
    
    return '\n'.join(latex)


def create_latex_table_for_scenario(comparison_df: pd.DataFrame, scenario: str) -> str:
    """
    Create a LaTeX table for a specific simulation scenario with ALL metrics
    
    Args:
        comparison_df: Multi-indexed DataFrame with comparison results
        scenario: Scenario name ('basic', 'variable_computational_budget', or 'variable_computational_demand')
        
    Returns:
        LaTeX table string with all metrics for the scenario
    """
    # Define simulation type display names
    sim_names = {
        'basic': 'Base',
        'variable_computational_budget': 'Budget',
        'variable_computational_demand': 'Demand'
    }
    
    # Get metrics for this specific scenario
    scenario_metrics = []
    for metric in comparison_df.index.get_level_values('metric_name').unique():
        try:
            if (scenario, metric) in comparison_df.index:
                scenario_metrics.append(metric)
        except:
            continue
    
    # Sort metrics alphabetically for consistent ordering
    scenario_metrics.sort()
    
    # Define metric display names (simplified for readability in large table)
    def format_metric_name(metric_name: str) -> str:
        """Convert metric name to a more readable format"""
        # Replace underscores with spaces and title case
        formatted = metric_name.replace('_', ' ').title()
        
        # First, handle specific complete metric names
        specific_replacements = {
            'Time All Slo Fulfilled Simultaneously Percent': 'Time All SLOs Met',
            'Maximum Consecutive Timesteps With Slo Violations': 'Max Consecutive SLO Violations',
            'Average Timesteps To Reach Stable Quality Configuration': 'Avg. Time to Stable Config',
            'Average Avg Global Processing Time Slo Value': 'Avg. Global Processing Time SLO Value',
            'Average Avg Worker Processing Time Slo Value': 'Avg. Worker Processing Time SLO Value',
            'Avg Global Processing Time Avg Violation Severity Factor': 'Global Processing Time Violation Severity',
            'Avg Worker Processing Time Avg Violation Severity Factor': 'Worker Processing Time Violation Severity',
            'Avg Global Processing Time Stability Coefficient Of Variation': 'Global Processing Time Stability CoV',
            'Avg Worker Processing Time Stability Coefficient Of Variation': 'Worker Processing Time Stability CoV',
            'Memory Usage Average Violation Severity Factor': 'Memory Usage Violation Severity',
            'Queue Size Average Violation Severity Factor': 'Queue Size Violation Severity',
            'Queue Size Avg Violation Severity Factor': 'Queue Size Violation Severity'
        }
        
        # Apply specific replacements first
        for old, new in specific_replacements.items():
            if formatted == old:
                formatted = new
                break
        else:
            # Apply general replacements only if no specific replacement was found
            general_replacements = {
                'Slo': 'SLO',
                'Coefficient Of Variation': 'CoV',
                'Fulfillment Rate Percent': 'Fulfillment Rate',
                'Average Overall Stream Quality Score': 'Avg. Stream Quality Score',
                'Average Memory Usage Slo Value': 'Avg. Memory Usage SLO Value',
                'Average Queue Size Slo Value': 'Avg. Queue Size SLO Value',
                'Average Slo Fulfillment Rate Percent': 'Avg. SLO Fulfillment Rate',
                'Average Slo Value Across All Slos': 'Avg. SLO Value (All SLOs)',
                'Global Processing Time Slo Fulfillment Rate Percent': 'Global Processing Time SLO Fulfillment Rate',
                'Memory Usage Slo Fulfillment Rate Percent': 'Memory Usage SLO Fulfillment Rate',
                'Queue Size Slo Fulfillment Rate Percent': 'Queue Size SLO Fulfillment Rate',
                'Worker Processing Time Slo Fulfillment Rate Percent': 'Worker Processing Time SLO Fulfillment Rate',
                'Memory Usage Stability Coefficient Of Variation': 'Memory Usage Stability CoV',
                'Queue Size Stability Coefficient Of Variation': 'Queue Size Stability CoV'
            }
            for old, new in general_replacements.items():
                formatted = formatted.replace(old, new)
            
            # Finally, replace any remaining "Average" with "Avg." to ensure consistency
            formatted = formatted.replace('Average ', 'Avg. ')
        
        return formatted
    
    # Start LaTeX table
    latex = []
    latex.append(r'\begin{table}[h!]')
    latex.append(r'\centering')
    latex.append(r'\small')  # Smaller font size
    latex.append(f'\\caption{{Complete Comparison of All Evaluation Metrics - {sim_names[scenario]} Scenario}}')
    latex.append(f'\\label{{tab:complete_results_{scenario}}}')
    latex.append(r'\begin{tabular}{@{}llll@{}}')
    latex.append(r'\toprule')
    latex.append(r'\textbf{Metric} & \textbf{AIF} & \textbf{Heuristic} & \textbf{$\Delta$} \\')
    latex.append(r'\midrule')
    
    # Process each metric for this scenario
    for metric in scenario_metrics:
        try:
            # Get the data for this simulation and metric
            row = comparison_df.loc[(scenario, metric)]
            aif_value = row['aif_agent']
            heuristic_value = row['heuristic']
            delta_value = row['delta']
            
            # Format values based on metric type
            if 'timesteps' in metric.lower() and 'consecutive' in metric.lower():
                # Integer values for consecutive timesteps
                aif_str = f"{int(aif_value)}"
                heuristic_str = f"{int(heuristic_value)}"
            elif 'timesteps' in metric.lower():
                # Format to 1 decimal place for other timesteps
                aif_str = f"{aif_value:.1f}"
                heuristic_str = f"{heuristic_value:.1f}"
            elif 'percent' in metric.lower() or 'rate' in metric.lower():
                # Format percentages to 3 decimal places
                aif_str = f"{aif_value:.3f}"
                heuristic_str = f"{heuristic_value:.3f}"
            elif 'score' in metric.lower():
                # Format scores to 3 decimal places
                aif_str = f"{aif_value:.3f}"
                heuristic_str = f"{heuristic_value:.3f}"
            elif 'factor' in metric.lower():
                # Format factors to 2 decimal places
                aif_str = f"{aif_value:.2f}"
                heuristic_str = f"{heuristic_value:.2f}"
            else:
                # Default formatting to 4 decimal places
                aif_str = f"{aif_value:.4f}"
                heuristic_str = f"{heuristic_value:.4f}"
            
            # Format delta with proper sign
            if delta_value >= 0:
                delta_str = f"+{delta_value:.2f}\\%"
            else:
                delta_str = f"{delta_value:.2f}\\%"
            
            # Format metric name for display
            metric_display = format_metric_name(metric)
            
            # Create the row
            latex.append(f"{metric_display} & {aif_str} & {heuristic_str} & {delta_str} \\\\")
            
        except KeyError:
            print(f"Warning: Metric '{metric}' not found for simulation '{scenario}'")
            continue
    
    # Close the table
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')
    
    return '\n'.join(latex)


def create_latex_tables_complete(comparison_df: pd.DataFrame) -> dict:
    """
    Create comprehensive LaTeX tables for each simulation scenario with ALL metrics
    
    Args:
        comparison_df: Multi-indexed DataFrame with comparison results
        
    Returns:
        Dictionary with scenario names as keys and LaTeX table strings as values
    """
    scenarios = ['basic', 'variable_computational_budget', 'variable_computational_demand']
    tables = {}
    
    for scenario in scenarios:
        tables[scenario] = create_latex_table_for_scenario(comparison_df, scenario)
    
    return tables


def save_latex_table(latex_content: str, filename: str) -> None:
    """Save the LaTeX table string to file"""
    # Save to the same directory as the CSV results
    latex_path = EvaluationUtils.get_consolidated_filepath(DirectoryType.OUTPUT, filename, "tex")
    EvaluationUtils.ensure_directory_exists(latex_path)
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {latex_path}")


def save_latex_table_with_path(latex_content: str, sim_type: str, suffix: str) -> None:
    """Save the LaTeX table string to file using the evaluation utils path structure"""
    # Create a proper filepath using the evaluation utils pattern
    # This will save to out/latex/{sim_type}/complete_{suffix}.tex
    from evaluation.simulation.simulation_type import SimulationType
    
    # Convert string to enum if needed
    if isinstance(sim_type, str):
        sim_type_enum = getattr(SimulationType, sim_type.upper())
    else:
        sim_type_enum = sim_type
    
    # Use a modified path structure for latex files
    base_dir = DirectoryType.OUTPUT.value
    latex_dir = os.path.join(base_dir, "latex")
    sim_dir = os.path.join(latex_dir, sim_type.lower())
    filename = f"complete_{suffix}.tex"
    latex_path = os.path.join(sim_dir, filename)
    
    # Ensure directory exists
    EvaluationUtils.ensure_directory_exists(latex_path)
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {latex_path}")


def generate_and_save_latex_tables(comparison_df: pd.DataFrame) -> None:
    """Generate and save both summary and complete LaTeX tables to files"""
    # Generate and save summary table (single table for all scenarios)
    latex_summary = create_latex_table_summary(comparison_df)
    save_latex_table(latex_summary, "agent_comparison_table_summary")
    
    # Generate and save complete tables (separate table for each scenario)
    complete_tables = create_latex_tables_complete(comparison_df)
    
    for scenario, latex_content in complete_tables.items():
        save_latex_table_with_path(latex_content, scenario, "metrics")
