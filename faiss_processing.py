import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_json(file_path: str) -> List[Dict]:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_embeddings(chunks: List[Dict]):
    """Generate embeddings for chunks."""
    texts = []
    metadata = []
    for chunk in chunks:
        # Extract content and links
        content = chunk["content"] if isinstance(chunk["content"], str) else " ".join(chunk["content"])
        links = chunk.get("metadata", {}).get("links", [])
        links = " ".join(links) if isinstance(links, list) else str(links)

        # Combine content and links
        full_content = f"{content} {links}".strip()
        texts.append(full_content)
        
        # Combine metadata with content
        combined_metadata = {
            "content": content,  # Include the content
            **chunk["metadata"]  # Merge existing metadata
        }
        metadata.append(combined_metadata)

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, metadata

def save_faiss_index(index, metadata: List[Dict], index_file: str, metadata_file: str):
    """Save FAISS index and metadata."""
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def add_chunks_to_faiss(chunks_file: str, index_file: str, metadata_file: str):
    """Process chunks and add them to FAISS index."""
    chunks = load_json(chunks_file)

    # Generate embeddings and metadata
    embeddings, metadata = generate_embeddings(chunks)

    # Initialize or load FAISS index
    dimension = embeddings.shape[1]
    try:
        index = faiss.read_index(index_file)
        print("Existing FAISS index loaded.")
    except:
        index = faiss.IndexFlatL2(dimension)
        print("New FAISS index created.")

    index.add(embeddings)

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        metadata = existing_metadata + metadata
    except:
        print("No existing metadata found. Creating new metadata file.")

    # Save index and metadata
    save_faiss_index(index, metadata, index_file, metadata_file)
    print(f"Stored {len(chunks)} chunks in FAISS.")

if __name__ == "__main__":
    chunks_file = "processed_chunks.json"  
    index_file = "faiss_index.bin"         
    metadata_file = "faiss_metadata.json" 

    add_chunks_to_faiss(chunks_file, index_file, metadata_file)
