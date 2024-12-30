from openai import OpenAI
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to format context from retrieved chunks
def format_context(retrieved_chunks: List[Dict]) -> str:
    """
    Combine related chunks into a single context string.
    """
    context = []
    for chunk in retrieved_chunks:
        chunk_content = chunk["metadata"].get("content", "No content available.")
        tag = chunk["metadata"].get("tag", "unknown")
        file = chunk["metadata"].get("file", "unknown")
        tag_hierarchy = chunk["metadata"].get("tag_hierarchy", "unknown")

        context.append(
            f"Tag: {tag}\n"
            f"File: {file}\n"
            f"Hierarchy: {tag_hierarchy}\n"
            f"Content: {chunk_content}\n"
        )
    return "\n---\n".join(context)

# Query the LLM with a structured prompt
def query_llm(context: str, question: str) -> str:
    """
    Use an LLM to generate an answer based on the context and question.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer with reference to the semantic structure of the document."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the actual model name you're using
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Post-process the response
def post_process_response(response: str, format_type: str = "plain_text") -> str:
    """
    Post-process the LLM response to extract references and reformat the output.
    """
    if format_type == "plain_text":
        return response
    elif format_type == "json":
        return json.dumps({"response": response}, indent=4)
    elif format_type == "html":
        return f"<html><body><p>{response.replace('\n', '<br>')}</p></body></html>"
    else:
        raise ValueError("Unsupported format type. Use 'plain_text', 'json', or 'html'.")

# Main execution
if __name__ == "__main__":
    # Example retrieved chunks
    retrieved_chunks = [
                            {
    "distance": 0.3904612064361572,
    "metadata": {
        "id": 48,
        "tag": "h2",
        "file": "ndap_chap_5_6.html",
        "tag_hierarchy": "[document]/h2",
        "links": None
    },
    "links": None
},
{
    "distance": 0.8308258056640625,
    "metadata": {
        "id": 56,
        "tag": "p",
        "file": "ndap_chap_5_7.html",
        "tag_hierarchy": "[document]/p",
        "links": None
    },
    "links": None
},
{
    "distance": 0.9005655646324158,
    "metadata": {
        "id": 58,
        "tag": "page-footnote",
        "file": "ndap_chap_5_7.html",
        "tag_hierarchy": "[document]/page-footnote",
        "links": None
    },
    "links": None
},
{
    "distance": 0.9849951267242432,
    "metadata": {
        "id": 44,
        "tag": "page-footnote",
        "file": "ndap_chap_5_5.html",
        "tag_hierarchy": "[document]/page/page-footnote",
        "links": None
    },
    "links": None
},
{
    "distance": 0.9912818670272827,
    "metadata": {
        "id": 55,
        "tag": "p",
        "file": "ndap_chap_5_7.html",
        "tag_hierarchy": "[document]/p",
        "links": None
    },
    "links": None
}
    ]
    
    # User query
    user_query = "tell me more about the path of green transition in India"

    # Generate context
    context = format_context(retrieved_chunks)

    # Query LLM
    llm_response = query_llm(context, user_query)

    # Post-process the response
    formatted_response = post_process_response(llm_response, format_type="json")

    # Print the final output
    print("Final Response:")
    print(formatted_response)
