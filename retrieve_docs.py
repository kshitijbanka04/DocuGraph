import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Utility Functions

def load_json(file_path: str) -> List[Dict]:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def generate_embeddings(chunks: List[Dict]):
    """Generate embeddings for chunks."""
    texts = []
    for chunk in chunks:
        content = chunk["content"] if isinstance(chunk["content"], str) else " ".join(chunk["content"])
        links = chunk.get("metadata", {}).get("links", [])
        links = " ".join(links) if isinstance(links, list) else str(links)
        texts.append(f"{content} {links}")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, texts

def save_faiss_index(index, metadata: List[Dict], index_file: str, metadata_file: str):
    """Save FAISS index and metadata."""
    faiss.write_index(index, index_file)
    save_json(metadata, metadata_file)

def load_faiss_index(index_file: str, metadata_file: str):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_file)
    metadata = load_json(metadata_file)
    return index, metadata

def store_chunks_in_faiss(chunks_file: str, index_file: str, metadata_file: str):
    """Process chunks and store in FAISS index."""
    chunks = load_json(chunks_file)
    embeddings, texts = generate_embeddings(chunks)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Prepare metadata
    metadata = [{"id": idx, **chunk["metadata"]} for idx, chunk in enumerate(chunks)]

    save_faiss_index(index, metadata, index_file, metadata_file)
    print(f"Stored {len(chunks)} chunks in FAISS.")

def query_faiss(index, metadata: List[Dict], query_embedding, k: int = 5):
    """Query FAISS index."""
    distances, indices = index.search(np.array([query_embedding]), k)
    return [
        {
            "distance": float(distances[0][i]),  # Convert numpy.float32 to float
            "metadata": metadata[indices[0][i]],
            "links": metadata[indices[0][i]].get("links", [])
        }
        for i in range(len(indices[0]))
    ]

# Main Execution
if __name__ == "__main__":
    chunks_file = "processed_chunks_with_links.json"  # File containing processed chunks
    index_file = "faiss_index.bin"
    metadata_file = "faiss_metadata.json"

    # Store chunks in FAISS index
    store_chunks_in_faiss(chunks_file, index_file, metadata_file)

    # Load FAISS index and metadata
    index, metadata = load_faiss_index(index_file, metadata_file)

    # Query FAISS
    query_text = "tell me more about the path of green transition in India"
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
    results = query_faiss(index, metadata, query_embedding, k=5)

    # Display results
    print("Query Results:")
    for result in results:
        print(json.dumps(result, indent=4))
