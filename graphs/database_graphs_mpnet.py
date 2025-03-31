import matplotlib.pyplot as plt

# Data for Memory Usage Differences
memory_stages = ["Redis Setup", "Milvus Setup", "Chroma Setup"]
memory_values = [464.86, 464.92, 490.41, 527.78]  # Memory at each stage

# Calculate memory difference compared to previous stages
memory_diff = [memory_values[i] - memory_values[i-1] for i in range(1, len(memory_values))]

# Setup times for each database
setup_times = [0.2411, 1.1047, 1.0181]  # Redis, Milvus, Chroma setup times

# Data for Query Times
query_stages = ["Milvus Query", "Redis Query", "Chroma Query"]
query_times = [0.0101, 0.0023, 0.0048]  # Query times for each

# Create subplots for three separate bar charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Memory Usage Differences
axes[0].bar(memory_stages, memory_diff, color='b')
axes[0].set_title("Memory Usage Differences (all-mpnet-base-v2)")
axes[0].set_xlabel("Stages")
axes[0].set_ylabel("Memory Difference (MB)")
axes[0].grid(True, axis='y')

# 2. Setup Times
axes[1].bar(memory_stages, setup_times, color='orange')
axes[1].set_title("Setup Times (all-mpnet-base-v2)")
axes[1].set_xlabel("Stages")
axes[1].set_ylabel("Setup Time (seconds)")
axes[1].grid(True, axis='y')

# 3. Query Times
axes[2].bar(query_stages, query_times, color='g')
axes[2].set_title("Query Times (all-mpnet-base-v2)")
axes[2].set_xlabel("Databases")
axes[2].set_ylabel("Query Time (seconds)")
axes[2].grid(True, axis='y')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()