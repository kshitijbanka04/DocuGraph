# DocuGraph

**DocuGraph** is a system for semantic search and question-answering based on XML-structured documents. It leverages **FAISS** as a vector database for efficient embedding storage and retrieval and uses **all-MiniLM-L6-v2** for generating embeddings due to its balance of performance and computational efficiency.

---

## Features

### 1. Preprocessing XML Documents
- Parses XML documents into meaningful chunks while retaining semantic structure.
- Extracts and stores metadata such as `tag_hierarchy` and `element_type` for contextual filtering.
- Supports paragraphs, tables, lists, and other content types.

### 2. Embedding Storage with FAISS
- Utilizes **FAISS** for storing vector embeddings efficiently.
- Metadata and raw content are stored alongside embeddings for enhanced query handling.

### 3. Query Enhancement
- User queries are enhanced with context using an LLM.
- Metadata filters (e.g., `tag_hierarchy`, `element_type`) refine results.

### 4. Semantic Retrieval and Answer Generation
- Retrieves relevant chunks based on semantic similarity and metadata.
- Generates structured answers, preserving XML semantics and providing contextual responses.

---

## Getting Started

### 1. Install Dependencies
Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```
Start the app with:

```bash
streamlit run streamlit_app.py
```

## Why all-MiniLM-L6-v2?
1.Compact yet Powerful: Suitable for environments with limited computational resources.
2.Balanced Performance: Achieves high-quality embeddings without heavy overhead.
3.Versatile: Performs well across diverse semantic search and retrieval tasks.

## File Structure
streamlit_app.py: Streamlit app for user queries and document interaction.
preprocessing.py: Handles XML parsing, chunking, and metadata extraction.
faiss_processing.py: Embedding generation, FAISS integration.
requirements.txt: List of dependencies for running the project.
