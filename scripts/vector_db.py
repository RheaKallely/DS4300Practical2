import numpy as np
import redis
import chromadb
from chromadb.config import Settings
from pymilvus import connections, Collection
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import ollama
import os

# === CONFIG ===
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_PATHS = [
    "data/embedded/all-MiniLM-L6-v2_embeddings.npy",
    "data/embedded/all-mpnet-base-v2_embeddings.npy",
    "data/embedded/InstructorXL_embeddings.npy"
]
TEXT_PATH = "data/process_text/processed_chunks.txt"
EMBEDDING_DIM = 384
REDIS_INDEX_NAME = "redis_MiniLM"
CHROMA_COLLECTION = "chroma_MiniLM"
MILVUS_COLLECTION = "milvus_MiniLM"
OLLAMA_MODEL = "tinyllama"
VECTOR_BACKEND = "milvus"  # This can be ignored, as we will query all three databases

# === Load Chunks and Embeddings ===
embeddings_list = [np.load(path) for path in VECTOR_PATHS]
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text_chunks = [line.strip() for line in f if line.strip() and not line.startswith("---")]

# === Load Embedding Model ===
embed_model = SentenceTransformer(f"sentence-transformers/{MODEL_NAME}")

# === Ask the User ===
question = input("❓ Ask a question: ")
query_vec = embed_model.encode(question)

# === Retrieval Functions ===

def query_redis(query_vec):
    r = redis.Redis(host="localhost", port=6379, db=0)
    q = (
        Query("*=>[KNN 3 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("text", "score")
        .dialect(2)
    )
    res = r.ft(REDIS_INDEX_NAME).search(q, query_params={"vec": np.array(query_vec, dtype=np.float32).tobytes()})
    
    if not res.docs:
        print("No results from Redis.")
    
    return [(doc.text, float(doc.score)) for doc in res.docs] if res.docs else []

def query_chroma(query_vec):
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    result = collection.query(query_embeddings=[query_vec], n_results=3)
    
    if not result['documents'][0]:
        print("No results from Chroma.")
    
    return [(doc, dist) for doc, dist in zip(result['documents'][0], result['distances'][0])] if result['documents'][0] else []

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
    
    if not results[0]:
        print("No results from Milvus.")
    
    return [(hit.entity.get("text"), hit.distance) for hit in results[0]] if results[0] else []

# === Query all databases ===
redis_results = query_redis(query_vec)
chroma_results = query_chroma(query_vec)
milvus_results = query_milvus(query_vec)

# === Combine Results from All Databases ===
all_results = redis_results + chroma_results + milvus_results

# === Check if Any Results Were Retrieved ===
if all_results:
    # === Sort Results by Score/Distance ===
    sorted_results = sorted(all_results, key=lambda x: x[1])  # Sorting by score/distance

    # === Extract Context for the Prompt ===
    context = "\n\n".join([chunk for chunk, _ in sorted_results])

    # === Build the Prompt for LLM ===
    prompt = f"""You are an assistant answering questions using the following course notes.

    NOTES:
    {context}

    QUESTION:
    {question}

    Answer:"""
else:
    prompt = f"""You are an assistant answering questions, but no relevant notes were retrieved.

    QUESTION:
    {question}

    Answer:"""

# === Run the LLM (Ollama) ===
try:
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    print("\n🤖 Answer:")
    print(answer)
except Exception as e:
    print(f"\n❌ Error running Ollama: {e}")

# === Save Results to a File ===
output_file = "retrieval_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Question: {question}\n\n")
    f.write(f"Top Chunks Retrieved from Redis:\n")
    for chunk, score in redis_results:
        f.write(f"{chunk}\nScore: {score}\n\n")
    
    f.write(f"Top Chunks Retrieved from Chroma:\n")
    for chunk, score in chroma_results:
        f.write(f"{chunk}\nScore: {score}\n\n")
    
    f.write(f"Top Chunks Retrieved from Milvus:\n")
    for chunk, score in milvus_results:
        f.write(f"{chunk}\nScore: {score}\n\n")
    
    f.write(f"Answer from LLM:\n{answer}\n")

print(f"\n✅ Results saved to {output_file}")
