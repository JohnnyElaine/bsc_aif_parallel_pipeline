import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from producer.elasticity.agent.reinforcement_learning_agent import ReinforcementLearningAgent

from producer.elasticity.handler.elasticity_handler import ElasticityHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    # Initialize your TaskConfig, TaskGenerator, and RequestHandler first (not shown)
    # ...

    # Create the elasticity handler
    elasticity_handler = ElasticityHandler(target_config, task_generator, request_handler)

    # Create the reinforcement learning agent
    agent = ReinforcementLearningAgent(elasticity_handler)

    # Option 1: Use a pre-trained model
    # agent = ReinforcementLearningAgent(elasticity_handler, model_path="./models/yolo_elasticity_agent.zip")

    # Option 2: Train the model first (this would typically be done offline)
    # agent.train(total_timesteps=50000)
    # agent.save("./models/yolo_elasticity_agent.zip")

    # Use the agent in your system
    try:
        # Run for a set period or until interrupted
        for i in range(1000):
            # Perform one step of the agent
            action, success = agent.step()

            # Optional: log the current state and action
            print(f"Step {i}: Action={action.name}, Success={success}")

            # Get current SLO status
            slo_stats = agent.get_slo_statistics()
            print(f"SLO Status: {slo_stats}")

            # Wait a bit before the next iteration (adjust as needed)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping the agent...")

    # After running, analyze the performance
    analyze_performance(agent)


def analyze_performance(agent):
    """Analyze and visualize the agent's performance"""
    # Get action statistics
    action_stats = agent.get_action_statistics()
    print("\nAction Statistics:")
    print(action_stats)

    # Plot action distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Action', y='Count', data=action_stats)
    plt.title('Action Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('action_distribution.png')

    # Get performance history
    history = agent.get_performance_history()

    # Plot SLO status over time
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    for status in ['queue_status', 'memory_status']:
        status_numeric = history[status].map({'OK': 0, 'WARNING': 1, 'CRITICAL': 2})
        plt.plot(status_numeric, label=status)
    plt.yticks([0, 1, 2], ['OK', 'WARNING', 'CRITICAL'])
    plt.legend()
    plt.title('SLO Status Over Time')

    # Plot reward over time
    plt.subplot(3, 1, 2)
    plt.plot(history['reward'])
    plt.title('Reward Over Time')

    # Plot quality parameters over time
    plt.subplot(3, 1, 3)
    plt.plot(pd.to_numeric(history['fps']), label='FPS')
    plt.plot(pd.to_numeric(history['workload'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2})), label='Workload')
    plt.legend()
    plt.title('Quality Parameters Over Time')

    plt.tight_layout()
    plt.savefig('performance_history.png')

    # Get SLO statistics
    slo_stats = agent.get_slo_statistics()
    print("\nSLO Statistics:")
    print(slo_stats)


if __name__ == "__main__":
    main()