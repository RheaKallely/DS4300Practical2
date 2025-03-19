import time
import psutil
from sentence_transformers import SentenceTransformer
import os

# Define the models for comparison
models = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "InstructorXL": "huggingface/instructor-xl",  # Example model from Huggingface
}

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Memory in MB

# Function to embed text using the given model
def embed_text_with_model(model_name, text_chunks):
    print(f"Embedding using model: {model_name}")
    
    try:
        # Load the model
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None  # Return None if the model fails to load

    embeddings = []
    
    # Measure the start time and memory usage
    start_time = time.time()
    start_memory = get_memory_usage()

    # Embedding all chunks
    for chunk in text_chunks:
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    
    # Measure the end time and memory usage
    end_time = time.time()
    end_memory = get_memory_usage()

    # Calculate the results
    embedding_time = end_time - start_time
    memory_used = end_memory - start_memory
    num_embeddings = len(embeddings)

    # Print out the results
    print(f"Embedding Time: {embedding_time:.2f} seconds")
    print(f"Memory Used: {memory_used:.2f} MB")
    print(f"Number of embeddings: {num_embeddings}")

    return embedding_time, memory_used, num_embeddings

# Save the results to a text file
def save_results_to_file(results, output_folder="test_results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, "embedding_results.txt")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for model_name, (embedding_time, memory_used, num_embeddings) in results.items():
                if embedding_time is not None:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Embedding Time: {embedding_time:.2f} seconds\n")
                    f.write(f"Memory Used: {memory_used:.2f} MB\n")
                    f.write(f"Number of embeddings: {num_embeddings}\n")
                    f.write("\n" + "-"*40 + "\n")
        
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

# Function to compare multiple models on the given text chunks
def compare_models_on_chunks(text_chunks):
    results = {}

    for model_name, model_path in models.items():
        embedding_time, memory_used, num_embeddings = embed_text_with_model(model_path, text_chunks)
        results[model_name] = (embedding_time, memory_used, num_embeddings)

    # Save results to a file
    save_results_to_file(results)

# Example usage: assume you have some text chunks from `process_notes_folder`
text_chunks = ["This is a sample chunk.", "Another chunk of text.", "Embedding testing with multiple models."]  # Replace with actual chunks

# Compare models and save results
compare_models_on_chunks(text_chunks)
