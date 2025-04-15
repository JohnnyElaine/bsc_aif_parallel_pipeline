import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.figure(figsize=(12, 5))

# Plot 1: Worker task distribution
plt.subplot(1, 2, 1)
workers = worker_stats.sort_values('num_requested_tasks', ascending=False)
sns.barplot(data=workers, x=workers.index, y='num_requested_tasks', palette='viridis')
plt.title('Tasks Requested per Worker')
plt.xlabel('Worker ID')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45)

# Plot 2: Registration time analysis
plt.subplot(1, 2, 2)
workers['registration_human'] = pd.to_datetime(workers['registration_time'], unit='s')
sns.scatterplot(data=workers, x='registration_human', y='num_requested_tasks',
                hue=workers.index, s=200)
plt.title('Registration Time vs Task Count')
plt.xlabel('Registration Time')
plt.ylabel('Tasks Requested')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()