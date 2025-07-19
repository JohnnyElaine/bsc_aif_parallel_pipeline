import json
import os
from typing import Dict, Any, List

# TODO make this into latex table
class MetricComparator:
    """Compare metrics between different agent types for each simulation type"""
    
    AGENT_TYPES = ['active_inference_relative_control', 'heuristic']
    SIMULATION_TYPES = ['basic', 'variable_computational_demand', 'variable_computational_budget']
    
    def __init__(self, metrics_dir: str = "out/metrics", output_dir: str = "out/comparison"):
        self.metrics_dir = metrics_dir
        self.output_dir = output_dir
    
    def compare_all_simulations(self):
        """Compare metrics between agent types for all simulation types"""
        for sim_type in self.SIMULATION_TYPES:
            self.compare_simulation_type(sim_type)
    
    def compare_simulation_type(self, sim_type: str):
        """Compare metrics between agent types for a specific simulation type"""
        # Load metrics for both agent types
        aif_metrics = self.load_metrics('active_inference_relative_control', sim_type)
        heuristic_metrics = self.load_metrics('heuristic', sim_type)
        
        if not aif_metrics or not heuristic_metrics:
            print(f"Warning: Could not load metrics for simulation type '{sim_type}'")
            return
        
        # Calculate comparison
        comparison = self.calculate_comparison(aif_metrics, heuristic_metrics, sim_type)
        
        # Save comparison results
        self.save_comparison(comparison, sim_type)
    
    def load_metrics(self, agent_type: str, sim_type: str) -> Dict[str, Any]:
        """Load metrics from JSON file for a specific agent type and simulation type"""
        filepath = os.path.join(self.metrics_dir, f'{sim_type}_sim', f'{agent_type}_metrics.json')
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Metrics file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON file {filepath}: {e}")
            return {}
    
    def calculate_comparison(self, aif_metrics: Dict[str, Any], heuristic_metrics: Dict[str, Any], sim_type: str) -> Dict[str, Any]:
        """
        Calculate percentage difference between AIF and Heuristic agent metrics
        
        Formula: ((AIF_value - Heuristic_value) / Heuristic_value) * 100
        Positive values mean AIF performs better (for metrics where higher is better)
        Negative values mean Heuristic performs better (for metrics where higher is better)
        """
        comparison = {
            'simulation_type': sim_type,
            'comparison_method': 'active_inference_relative_control vs heuristic',
            'formula': '((aif_value - heuristic_value) / heuristic_value) * 100',
            'interpretation': {
                'positive_values': 'AIF agent has higher value than Heuristic agent',
                'negative_values': 'Heuristic agent has higher value than AIF agent',
                'note': 'Interpretation depends on whether higher values are better for each metric'
            },
            'metric_deltas_percent': {}
        }
        
        # Get all numeric metrics (excluding metadata)
        excluded_keys = {'agent_type', 'simulation_type'}
        
        for key in aif_metrics:
            if key in excluded_keys:
                continue
                
            aif_value = aif_metrics.get(key)
            heuristic_value = heuristic_metrics.get(key)
            
            # Skip if either value is missing or not numeric
            if (aif_value is None or heuristic_value is None or 
                not isinstance(aif_value, (int, float)) or 
                not isinstance(heuristic_value, (int, float))):
                continue
            
            # Calculate percentage difference
            if heuristic_value != 0:
                delta_percent = ((aif_value - heuristic_value) / heuristic_value) * 100
                comparison['metric_deltas_percent'][key] = round(delta_percent, 4)
            else:
                # Handle division by zero
                if aif_value == 0:
                    comparison['metric_deltas_percent'][key] = 0.0
                else:
                    comparison['metric_deltas_percent'][key] = float('inf') if aif_value > 0 else float('-inf')
        
        # Add summary statistics
        deltas = [v for v in comparison['metric_deltas_percent'].values() 
                 if v != float('inf') and v != float('-inf')]
        
        if deltas:
            comparison['summary'] = {
                'total_metrics_compared': len(deltas),
                'average_delta_percent': round(sum(deltas) / len(deltas), 4),
                'metrics_where_aif_better': sum(1 for d in deltas if d > 0),
                'metrics_where_heuristic_better': sum(1 for d in deltas if d < 0),
                'metrics_equal': sum(1 for d in deltas if d == 0)
            }
        
        # Add absolute values for easy reference
        comparison['absolute_values'] = {
            'active_inference_relative_control': {k: v for k, v in aif_metrics.items() if k not in excluded_keys},
            'heuristic': {k: v for k, v in heuristic_metrics.items() if k not in excluded_keys}
        }
        
        return comparison
    
    def save_comparison(self, comparison: Dict[str, Any], sim_type: str):
        """Save comparison results to JSON file"""
        # Create output directory
        dir_path = os.path.join(self.output_dir, f'{sim_type}_sim')
        os.makedirs(dir_path, exist_ok=True)
        
        # Create filename following the same pattern
        filename = f'aif_vs_heuristic_comparison.json'
        filepath = os.path.join(dir_path, filename)
        
        # Save with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Saved comparison for {sim_type} simulation to: {filepath}")
        
        # Print summary to console
        if 'summary' in comparison:
            summary = comparison['summary']
            print(f"  - Total metrics compared: {summary['total_metrics_compared']}")
            print(f"  - Average delta: {summary['average_delta_percent']:.2f}%")
            print(f"  - AIF better: {summary['metrics_where_aif_better']}, Heuristic better: {summary['metrics_where_heuristic_better']}")
    
    def generate_overall_comparison_report(self):
        """Generate an overall comparison report across all simulation types"""
        overall_report = {
            'comparison_type': 'Overall Agent Performance Comparison',
            'agent_types': self.AGENT_TYPES,
            'simulation_types': self.SIMULATION_TYPES,
            'simulation_comparisons': {}
        }
        
        for sim_type in self.SIMULATION_TYPES:
            comparison_file = os.path.join(self.output_dir, f'{sim_type}_sim', 'aif_vs_heuristic_comparison.json')
            try:
                with open(comparison_file, 'r') as f:
                    sim_comparison = json.load(f)
                    if 'summary' in sim_comparison:
                        overall_report['simulation_comparisons'][sim_type] = sim_comparison['summary']
            except FileNotFoundError:
                print(f"Warning: Comparison file not found for {sim_type}")
        
        # Save overall report
        os.makedirs(self.output_dir, exist_ok=True)
        overall_filepath = os.path.join(self.output_dir, 'overall_agent_comparison_report.json')
        with open(overall_filepath, 'w') as f:
            json.dump(overall_report, f, indent=2)
        
        print(f"Saved overall comparison report to: {overall_filepath}")


def compare_agent_metrics():
    """
    Convenience function to compare all agent metrics across simulation types
    """
    comparator = MetricComparator()
    print("Starting agent metric comparison...")
    
    comparator.compare_all_simulations()
    comparator.generate_overall_comparison_report()
    
    print("Agent metric comparison completed!")


if __name__ == "__main__":
    compare_agent_metrics()
