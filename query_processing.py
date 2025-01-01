from typing import List

def expand_query(query: str, synonyms: dict) -> str:
    expanded_query = query.lower()
    for term, synonym_list in synonyms.items():
        if term in expanded_query:
            expanded_query += " " + " ".join(synonym_list)
    return expanded_query

import json

# Load the synonyms dictionary from the JSON file
def load_synonyms(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        synonyms_dict = json.load(file)
    return synonyms_dict

file_path = "synonyms.json"

user_query = "What are India's strategies for the green transition?"

expanded_query = expand_query(user_query, load_synonyms(file_path))
print("Expanded Query:", expanded_query)
