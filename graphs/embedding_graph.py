import matplotlib.pyplot as plt

# Data
data = {
    "all-MiniLM-L6-v2": {
        "embedding_time": 15.72,
        "memory_used_MB": 136.59,
        "num_embeddings": 1338
    },
    "all-mpnet-base-v2": {
        "embedding_time": 30.55,
        "memory_used_MB": 328.28,
        "num_embeddings": 1338
    },
    "InstructorXL": {
        "embedding_time": 215.70,
        "memory_used_MB": 389.5,
        "num_embeddings": 1338
    }
}

models = list(data.keys())
embedding_times = [data[model]["embedding_time"] for model in models]
memory_usages = [data[model]["memory_used_MB"] for model in models]

# Plot 1: Embedding time
plt.figure(figsize=(8, 5))
plt.bar(models, embedding_times)
plt.title("Embedding Time per Model")
plt.xlabel("Model")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.show()

# Plot 2: Memory usage
plt.figure(figsize=(8, 5))
plt.bar(models, memory_usages)
plt.title("Memory Usage per Model")
plt.xlabel("Model")
plt.ylabel("Memory Used (MB)")
plt.tight_layout()
plt.show()