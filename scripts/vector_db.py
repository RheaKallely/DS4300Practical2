import numpy as np
import redis
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import ollama

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
OLLAMA_MODEL = "mistral"
VECTOR_BACKEND = "chroma"

# Load chunks and embeddings
embeddings_list = [np.load(path) for path in VECTOR_PATHS]
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text_chunks = [line.strip() for line in f if line.strip() and not line.startswith("---")]

# Load embedding model
embed_model = SentenceTransformer(f"sentence-transformers/{MODEL_NAME}")

connections.connect("default", host="localhost", port="19530")
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def setup_milvus(model_name: str):
    """Sets up Milvus: Replaces existing data if the collection exists, otherwise creates a new collection."""
    sanitized_model_name = model_name.replace("-", "_")
    embedding_file = f"data/embedded/{model_name}_embeddings.npy"
    embeddings = np.load(embedding_file, allow_pickle=True).tolist()

    connections.connect(host="localhost", port="19530")

    if utility.has_collection(sanitized_model_name):
        collection = Collection(sanitized_model_name)
        collection.drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
    ]
    schema = CollectionSchema(fields, description="Embeddings collection")
    collection = Collection(name=sanitized_model_name, schema=schema)

    collection.create_index(field_name="embedding", index_params={
        "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}
    })

    ids = list(range(len(embeddings)))
    data_to_insert = [ids, embeddings]
    collection.insert(data_to_insert)

    collection.load()
    print("Milvus setup completed successfully")
    return collection

def setup_redis(model_name: str):
    """Sets up Redis with vector search index."""
    embedding_file = f"data/embedded/{model_name}_embeddings.npy"
    embeddings = np.load(embedding_file, allow_pickle=True).tolist()

    dummy_texts = [f"Document {i} text" for i in range(len(embeddings))]

    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    # Clears existing data
    r.flushdb()

    pipeline = r.pipeline()
    for idx, (emb, text) in enumerate(zip(embeddings, dummy_texts)):
        key = f"{model_name}:{idx}"
        pipeline.hset(key, mapping={
            "embedding": np.array(emb, dtype=np.float32).tobytes(),
            "text": text.encode("utf-8")
        })
    pipeline.execute()

    try:
        r.ft(REDIS_INDEX_NAME).info()
    except redis.exceptions.ResponseError:
        r.ft(REDIS_INDEX_NAME).create_index([
            redis.commands.search.field.TagField("id"),
            redis.commands.search.field.VectorField(
                "embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"}
            ),
            redis.commands.search.field.TextField("text")
        ])

    print("Redis setup completed successfully")
    return r




def setup_chroma(model_name: str):
    """Sets up ChromaDB by clearing the existing collection (if data exists) and re-indexing the embeddings."""
    client = chromadb.HttpClient(host="localhost", port=8000)  # Connect to ChromaDB running in Docker
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Check if collection contains existing data
    existing_docs = collection.get()  # Retrieve existing documents
    if existing_docs and "ids" in existing_docs and existing_docs["ids"]:
        collection.delete(ids=existing_docs["ids"])  # Delete all documents by ID

    # Load the embeddings and index them
    embedding_file = f"data/embedded/{model_name}_embeddings.npy"
    embeddings = np.load(embedding_file, allow_pickle=True).tolist()

    ids = [str(i) for i in range(len(embeddings))]  # Ensure IDs are strings
    metadata = [{"index": i} for i in range(len(embeddings))]

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadata)
    print("Chroma setup completed successfully")

    return collection



# Initialize the databases
model = MODEL_NAME
milvus_collection = setup_milvus(model)
redis_client = setup_redis(model)
chroma_collection = setup_chroma(model)

# Retrieval functions
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

# Continuous question loop
while True:
    question = input("❓ Ask a question (or type 'exit' to quit): ")
   
    if question.lower() == 'exit':
        print("Goodbye!")
        break
   
    query_vec = embed_model.encode(question)

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

    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    prompt = f"""You are an assistant answering questions using the following course notes.

    NOTES:
    {context}

    QUESTION:
    {question}

    Answer:"""

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        print("\n🤖 Answer:")
        print(response["message"]["content"])
    except Exception as e:
        print(f"\n❌ Error running Ollama: {e}")
