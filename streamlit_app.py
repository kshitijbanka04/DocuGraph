import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from generate_answers_test import query_llm, format_context
from dotenv import load_dotenv
import os

# Load environment variables for OpenAI API
load_dotenv()

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
def load_faiss_index(index_file: str, metadata_file: str):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# Query FAISS
def query_faiss(index, metadata: List[Dict], query: str, k: int = 5):
    """Query FAISS index."""
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    results = [
        {
            "distance": float(distances[0][i]),
            "metadata": metadata[indices[0][i]],
            "links": metadata[indices[0][i]].get("links", []),
            "content": metadata[indices[0][i]].get("content", "")
        }
        for i in range(len(indices[0]))
    ]
    return results

# Streamlit App
def main():
    # Load FAISS index and metadata
    index_file = "faiss_index.bin"
    metadata_file = "faiss_metadata.json"
    index, metadata = load_faiss_index(index_file, metadata_file)

    st.title("🤖 Intelligent Document Search and Answer Generator")
    st.write("Input a query to retrieve relevant documents and generate an answer.")

    # Conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    user_query = st.text_input("Enter your query:")

    if st.button("Generate Answer") and user_query:
        # Step 1: Retrieve relevant documents
        st.write("Fetching relevant documents...")
        retrieved_chunks = query_faiss(index, metadata, user_query, k=5)

        # Step 2: Format retrieved chunks into context
        context = format_context(retrieved_chunks)

        # Step 3: Enhance query and generate an answer using LLM
        st.write("Generating answer using GPT...")
        conversation = st.session_state.conversation_history + [
            {"role": "user", "content": user_query}
        ]
        llm_response = query_llm(context, user_query)

        # Save conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_query})
        st.session_state.conversation_history.append({"role": "assistant", "content": llm_response})

        # Step 4: Display results
        st.subheader("🔍 Relevant Documents")
        for chunk in retrieved_chunks:
            with st.expander(f"Distance: {chunk['distance']:.4f}"):
                st.markdown(f"**Content:** {chunk['metadata'].get('content', 'No content available')}")
                if chunk["links"]:
                    st.markdown(f"**Links:** {', '.join(chunk['links'])}")

        st.subheader("🎉 Final Answer")
        st.markdown(f"""
            <div style="
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 10px; 
                border: 1px solid #ddd; 
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
                font-size: 16px; 
                color: #333;">
                <strong>{llm_response}</strong>
            </div>
        """, unsafe_allow_html=True)


        st.subheader("🔍 Conversation History")
        for message in st.session_state.conversation_history:
            role = "You" if message["role"] == "user" else "Assistant"
            with st.container():
                st.markdown(f"<div style='margin:5px 0;'><strong>{role}:</strong> {message['content']}</div>", 
                            unsafe_allow_html=True)

if __name__ == "__main__":
    main()
