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
VECTOR_BACKEND = "milvus"  # Change to: "redis", "milvus", or "chroma"

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
    return [(doc.text, float(doc.score)) for doc in res.docs]

def query_chroma(query_vec):
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    result = collection.query(query_embeddings=[query_vec], n_results=3)
    return [(doc, dist) for doc, dist in zip(result['documents'][0], result['distances'][0])]

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

# === Select DB and Query ===
if VECTOR_BACKEND == "redis":
    print("\n🔍 Using Redis for retrieval...")
    top_chunks = query_redis(query_vec)
elif VECTOR_BACKEND == "chroma":
    print("\n🔍 Using Chroma for retrieval...")
    top_chunks = query_chroma(query_vec)
elif VECTOR_BACKEND == "milvus":
    print("\n🔍 Using Milvus for retrieval...")
    top_chunks = query_milvus(query_vec)
else:
    raise ValueError("Unsupported VECTOR_BACKEND. Choose redis, chroma, or milvus.")

# === Build Prompt ===
context = "\n\n".join([chunk for chunk, _ in top_chunks])
prompt = f"""You are an assistant answering questions using the following course notes.

NOTES:
{context}

QUESTION:
{question}

Answer:"""

# === Run LLM ===
try:
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    print("\n🤖 Answer:")
    print(response["message"]["content"])
except Exception as e:
    print(f"\n❌ Error running Ollama: {e}")
