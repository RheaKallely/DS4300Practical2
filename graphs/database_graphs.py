import matplotlib.pyplot as plt

# Data
data = {
    "REDIS": {
        "latency": 0.0336,
        "memory_used_MB": 0.03,
        "top_chunks": []
    },
    "CHROMA": {
        "latency": 1.8672,
        "memory_used_MB": 2.43,
        "top_chunks": []
    },
    "MILVUS": {
        "latency": 0.3179,
        "memory_used_MB": 3.88,
        "top_chunks": [("Collections - a set of records of the same entity type - a table", 0.8209)]
    }
}

dbs = list(data.keys())
latencies = [data[db]["latency"] for db in dbs]
memory_usages = [data[db]["memory_used_MB"] for db in dbs]

# Plot 1: Latency
plt.figure(figsize=(8, 5))
plt.bar(dbs, latencies)
plt.title("Query Latency by Vector DB")
plt.xlabel("Database")
plt.ylabel("Latency (seconds)")
plt.tight_layout()
plt.show()

# Plot 2: Memory usage
plt.figure(figsize=(8, 5))
plt.bar(dbs, memory_usages)
plt.title("Memory Usage by Vector DB")
plt.xlabel("Database")
plt.ylabel("Memory Used (MB)")
plt.tight_layout()
plt.show()