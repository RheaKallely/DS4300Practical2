import time
import psutil
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

# Define the models for comparison
models = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "InstructorXL": "hkunlp/instructor-xl",  
}

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  

# Function to read text chunks from file
def load_text_chunks(file_path="data/process_text/processed_chunks.txt"):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = [line.strip() for line in file.readlines() if line.strip()]
    
    print(f"Loaded {len(chunks)} text chunks from {file_path}")
    return chunks

# Function to embed text using the given model
def embed_text_with_model(model_name, model_path, text_chunks):
    print(f"Embedding using model: {model_name}")
    
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None, None 

    embeddings = []
    
    # Measure the start time and memory usage
    start_time = time.time()
    start_memory = get_memory_usage()

    # Embed all chunks
    for chunk in text_chunks:
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    
    # Measure the end time and memory usage
    end_time = time.time()
    end_memory = get_memory_usage()

    # Calculate results
    embedding_time = end_time - start_time
    memory_used = end_memory - start_memory
    num_embeddings = len(embeddings)

    print(f"Embedding Time: {embedding_time:.2f} seconds")
    print(f"Memory Used: {memory_used:.2f} MB")
    print(f"Number of embeddings: {num_embeddings}")

    return embedding_time, memory_used, num_embeddings, embeddings

# Save the results to a JSON file 
def save_results_to_file(results, output_folder="test_results/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_json = os.path.join(output_folder, "embedding_results.json")
    
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results successfully saved to {output_json}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

# Save embeddings 
def save_embeddings_to_npy(embeddings, model_name, output_folder="data/embedded/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    npy_file = os.path.join(output_folder, f"{model_name}_embeddings.npy")
    np.save(npy_file, np.array(embeddings))
    print(f"Embeddings saved to {npy_file}")

# Function to compare multiple models on text chunks
def compare_models_on_chunks(text_chunks):
    results = {}

    for model_name, model_path in models.items():
        embedding_time, memory_used, num_embeddings, embeddings = embed_text_with_model(model_name, model_path, text_chunks)
        
        if embedding_time is not None:
            results[model_name] = {
                "embedding_time": embedding_time,
                "memory_used_MB": memory_used,
                "num_embeddings": num_embeddings
            }
            save_embeddings_to_npy(embeddings, model_name)

    save_results_to_file(results)

# Load text chunks from file
text_chunks = load_text_chunks("data/process_text/processed_chunks.txt")

# Run comparison if there are text chunks
if text_chunks:
    compare_models_on_chunks(text_chunks)
else:
    print("No text chunks found. Please check 'processed_chunks.txt'.")
