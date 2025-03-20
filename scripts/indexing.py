import numpy as np
import redis
import chromadb
from chromadb.config import Settings
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import ollama
import os
import time
import psutil
from scipy.spatial.distance import cosine

# === CONFIG ===
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_PATHS = [
    "data/embedded/all-MiniLM-L6-v2_embeddings.npy",
    "data/embedded/all-mpnet-base-v2_embeddings.npy",
    "data/embedded/InstructorXL_embeddings.npy"
]
TEXT_PATH = "data/process_text/processed_chunks.txt"
REDIS_INDEX_NAME = "redis_MiniLM"  # Not used anymore (no FT.SEARCH)
CHROMA_COLLECTION = "chroma_MiniLM"
MILVUS_COLLECTION = "milvus_MiniLM"
OLLAMA_MODEL = "tinyllama"
TEST_RESULTS_DIR = "test_results"

# Ensure test_results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# === Load Chunks and Embeddings ===
embeddings_list = [np.load(path) for path in VECTOR_PATHS]
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text_chunks = [line.strip() for line in f if line.strip() and not line.startswith("---")]

# === Load Embedding Model ===
embed_model = SentenceTransformer(f"sentence-transformers/{MODEL_NAME}")

# === Ask the User ===
question = input("❓ Ask a question: ")
query_vec = embed_model.encode(question)

# === Function to Measure Speed & Memory ===
def measure_performance(func, *args):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 ** 2  # in MB
    
    result = func(*args)
    
    mem_after = process.memory_info().rss / 1024 ** 2  # in MB
    elapsed_time = time.time() - start_time
    memory_used = mem_after - mem_before
    
    return result, elapsed_time, memory_used

# === Redis Retrieval (Without FT.SEARCH) ===
def query_redis(query_vec):
    r = redis.Redis(host="localhost", port=6379, db=0)

    # Retrieve all stored vectors from Redis
    keys = r.keys("doc:*")  # Assuming all embeddings are stored as "doc:<id>"
    
    similarities = []
    for key in keys:
        embedding = np.frombuffer(r.get(key), dtype=np.float32)
        score = 1 - cosine(query_vec, embedding)  # Cosine similarity
        text = r.hget(key, "text").decode("utf-8")
        similarities.append((text, score))
    
    # Return top 3 results
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

# === Chroma Retrieval ===
def query_chroma(query_vec):
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    result = collection.query(query_embeddings=[query_vec], n_results=3)
    return [(doc, dist) for doc, dist in zip(result['documents'][0], result['distances'][0])]

# === Milvus Retrieval ===
def query_milvus(query_vec):
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    results = collection.search(
        [query_vec],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["text"]
    )
    return [(hit.entity.get("text"), hit.distance) for hit in results[0]]

# === Run Tests ===
results = {}
for backend, query_func in zip(["redis", "chroma", "milvus"], [query_redis, query_chroma, query_milvus]):
    print(f"\n🔍 Testing {backend.capitalize()}...")
    top_chunks, latency, memory = measure_performance(query_func, query_vec)
    results[backend] = {"latency": latency, "memory_usage": memory, "results": top_chunks}

# === Save Results ===
results_path = os.path.join(TEST_RESULTS_DIR, "performance_results.txt")
with open(results_path, "w") as f:
    for backend, data in results.items():
        f.write(f"{backend.upper()}\n")
        f.write(f"Latency: {data['latency']:.4f} sec\n")
        f.write(f"Memory Used: {data['memory_usage']:.2f} MB\n")
        f.write(f"Top Chunks: {data['results']}\n\n")

print(f"\n✅ Test results saved to {results_path}")
