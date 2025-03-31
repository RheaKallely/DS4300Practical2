import numpy as np
import time
import psutil
import redis
import chromadb
import os
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAMES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "InstructorXL": "hkunlp/instructor-xl"
}

VECTOR_PATHS = [
    "data/embedded/all-MiniLM-L6-v2_embeddings.npy",
    "data/embedded/all-mpnet-base-v2_embeddings.npy",
    "data/embedded/InstructorXL_embeddings.npy"
]

def get_query_vector(model_name, embedding_dim):
    """Encodes a query using the correct model and ensures correct embedding dimension."""
    if model_name not in MODEL_NAMES:
        print(f"Skipping {model_name} due to missing model reference.")
        return None

    try:
        model = SentenceTransformer(MODEL_NAMES[model_name])
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    query_vec = model.encode("Sample query text")
    query_vec = np.array(query_vec, dtype=np.float32)

    if query_vec.shape[0] > embedding_dim:
        query_vec = query_vec[:embedding_dim]  # Truncate if too large
    elif query_vec.shape[0] < embedding_dim:
        query_vec = np.pad(query_vec, (0, embedding_dim - query_vec.shape[0]))  # Pad if too small

    return query_vec

def setup_milvus(model_name: str, embeddings: np.ndarray):
    """Sets up Milvus dynamically based on embedding dimensions."""
    sanitized_model_name = model_name.replace("-", "_")
    connections.connect(host="localhost", port="19530")

    if utility.has_collection(sanitized_model_name):
        Collection(sanitized_model_name).drop()

    embedding_dim = embeddings.shape[1]

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields, description="Embeddings collection")
    collection = Collection(name=sanitized_model_name, schema=schema)

    data_to_insert = [list(range(len(embeddings))), embeddings.astype(np.float32).tolist()]
    collection.insert(data_to_insert)

    collection.create_index(field_name="embedding", index_params={
        "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}
    })
   
    collection.load()
    print(f"Milvus setup completed for {model_name} with dimension {embedding_dim}")
    return collection

def setup_redis(model_name: str, embeddings: np.ndarray):
    """Sets up Redis dynamically for different embedding dimensions."""
    embedding_dim = embeddings.shape[1]
    index_name = f"redis_{embedding_dim}D"
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

    try:
        r.ft(index_name).info()
        print(f"Redis index {index_name} already exists.")
    except redis.exceptions.ResponseError:
        print(f"Creating Redis index {index_name}...")
        schema = (
            TextField("text"),
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": embedding_dim,
                "DISTANCE_METRIC": "COSINE"
            })
        )
        r.ft(index_name).create_index(
            schema, definition=IndexDefinition(prefix=[f"{model_name}:"], index_type=IndexType.HASH)
        )

    for idx, emb in enumerate(embeddings):
        key = f"{model_name}:{idx}"
        r.hset(key, mapping={
            "embedding": np.array(emb, dtype=np.float32).tobytes(),
            "text": f"Document {idx} text".encode("utf-8")
        })

    print(f"Redis setup completed for {index_name}")
    return r

def setup_chroma(model_name: str, embeddings: np.ndarray):
    """Sets up ChromaDB dynamically."""
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(name=model_name)

    existing_docs = collection.get()
    if existing_docs and "ids" in existing_docs and existing_docs["ids"]:
        collection.delete(ids=existing_docs["ids"])

    ids = [str(i) for i in range(len(embeddings))]
    metadata = [{"index": i} for i in range(len(embeddings))]
    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadata)

    print(f"Chroma setup completed for {model_name}")
    return collection

def test_query_speed(milvus_collection, redis_client, chroma_collection, query_vec, embedding_dim):
    """Tests query speed for Milvus, Redis, and Chroma."""
    results = "\nStarting Query Speed Test...\n"

    # Milvus Query
    milvus_start_time = time.time()
    milvus_collection.search(
        [query_vec], anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}}, limit=3
    )
    results += f"Milvus Query Time: {time.time() - milvus_start_time:.4f} seconds\n"

    # Redis Query
    redis_start_time = time.time()
    query = Query("*=>[KNN 3 @embedding $vec AS score]").sort_by("score").dialect(2)
    redis_client.ft(f"redis_{embedding_dim}D").search(query, query_params={"vec": np.array(query_vec, dtype=np.float32).tobytes()})
    results += f"Redis Query Time: {time.time() - redis_start_time:.4f} seconds\n"

    # Chroma Query
    chroma_start_time = time.time()
    chroma_collection.query(query_embeddings=[query_vec], n_results=3)
    results += f"Chroma Query Time: {time.time() - chroma_start_time:.4f} seconds\n"

    return results

def test_memory_usage(model_name, embeddings):
    """Tests memory usage before and after indexing and during database setup/querying."""
    results = f"\nStarting Memory Usage Test for {model_name}...\n"
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    results += f"Memory Before Setup: {mem_before:.2f} MB\n"

    embedding_dim = embeddings.shape[1]

    # Track memory usage during Redis setup
    redis_start_time = time.time()
    redis_client = setup_redis(model_name, embeddings)
    mem_after_redis = process.memory_info().rss / 1024 / 1024
    results += f"Memory After Redis Setup: {mem_after_redis:.2f} MB (Redis Setup Time: {time.time() - redis_start_time:.4f} seconds)\n"

    # Track memory usage during Milvus setup
    milvus_start_time = time.time()
    milvus_collection = setup_milvus(model_name, embeddings)
    mem_after_milvus = process.memory_info().rss / 1024 / 1024
    results += f"Memory After Milvus Setup: {mem_after_milvus:.2f} MB (Milvus Setup Time: {time.time() - milvus_start_time:.4f} seconds)\n"

    # Track memory usage during Chroma setup
    chroma_start_time = time.time()
    chroma_collection = setup_chroma(model_name, embeddings)
    mem_after_chroma = process.memory_info().rss / 1024 / 1024
    results += f"Memory After Chroma Setup: {mem_after_chroma:.2f} MB (Chroma Setup Time: {time.time() - chroma_start_time:.4f} seconds)\n"

    # Test querying
    query_vec = get_query_vector(model_name, embedding_dim)
    if query_vec is None:
        return results + "Skipping query tests due to missing model.\n"

    query_start_time = time.time()
    results += test_query_speed(milvus_collection, redis_client, chroma_collection, query_vec, embedding_dim)
    results += f"Query Test Time: {time.time() - query_start_time:.4f} seconds\n"

    # Final memory after query test
    mem_after_query = process.memory_info().rss / 1024 / 1024
    results += f"Memory After Query Test: {mem_after_query:.2f} MB\n"
    results += f"Total Memory Usage During Test: {mem_after_query - mem_before:.2f} MB\n"

    return results

def write_results_to_file(results, model_name):
    """Saves test results to a file."""
    file_path = f"test_results/{model_name}_results.txt"
    os.makedirs("test_results", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(results)
    print(f"Test results saved to {file_path}")

def run_tests():
    """Runs tests for each embedding model dynamically handling dimensions."""
    for path in VECTOR_PATHS:
        try:
            embeddings = np.load(path)
            model_name = os.path.basename(path).replace("_embeddings.npy", "")
            results = test_memory_usage(model_name, embeddings)
            write_results_to_file(results, model_name)
        except Exception as e:
            print(f"Error with {path}: {e}")

run_tests()