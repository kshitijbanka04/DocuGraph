import os
from bs4 import BeautifulSoup
import json
from typing import List, Dict

def parse_html(file_path: str) -> List[Dict]:
    """
    Parses an HTML/XML file and extracts structured chunks based on semantic units, 
    including hierarchical metadata and handling relational data like tables and links.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    chunks = []
    # Process all specified XML tags
    for tag in soup.find_all([
        'chapter-title', 'h1', 'h2', 'p', 'ol', 'ul', 'list-item',
        'page-module', 'figure-label', 'table', 'page-footer', 
        'page-header', 'caption', 'page-footnote', 'toc', 'toc-item'
    ]):
        # Identify type of content
        if tag.name in ['chapter-title', 'h1', 'h2']:
            chunk_type = "header"
        elif tag.name == 'p':
            chunk_type = "paragraph"
        elif tag.name in ['ol', 'ul']:
            chunk_type = "list"
        elif tag.name == 'table':
            chunk_type = "table"
        elif tag.name in ['page-header', 'page-footer', 'page-footnote']:
            chunk_type = "metadata"
        elif tag.name in ['toc', 'toc-item']:
            chunk_type = "toc"
        else:
            chunk_type = "other"

        # Process table content separately
        if tag.name == 'table':
            rows = []
            for row in tag.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                rows.append(cells)
            content = rows
        elif tag.name in ['ol', 'ul']:
            content = [li.get_text(strip=True) for li in tag.find_all('list-item')]
        else:
            content = ' '.join(tag.stripped_strings)

        # Extract links from within text or attributes
        links = []
        for a in tag.find_all('a', href=True):
            links.append(a['href'])
        # Check for URLs in text content if it is a string or a list
        if isinstance(content, str):
            links_in_text = [word for word in content.split() if word.startswith("http")]
            links.extend(links_in_text)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    links_in_text = [word for word in item.split() if word.startswith("http")]
                    links.extend(links_in_text)

        if content or links:
            chunks.append({
                "type": chunk_type,
                "content": content if content else None,
                "metadata": {
                    "tag": tag.name,
                    "file": os.path.basename(file_path),
                    "tag_hierarchy": get_tag_hierarchy(tag),
                    "links": links if links else None  # Include links if they exist
                }
            })

    return chunks

def get_tag_hierarchy(tag):
    """
    Returns the hierarchical path of the given tag in the document.
    """
    hierarchy = []
    while tag:
        hierarchy.insert(0, tag.name)
        tag = tag.parent
    return '/'.join(hierarchy)

# Parse all files
file_paths = [
    "ndap_chap_5_1.html",
    "ndap_chap_5_2.html",
    "ndap_chap_5_3.html",
    "ndap_chap_5_4.html",
    "ndap_chap_5_5.html",
    "ndap_chap_5_6.html",
    "ndap_chap_5_7.html",
    "ndap_chap_5_8.html",
    "ndap_chap_5_9.html",
    "ndap_chap_5_10.html"
]

all_chunks = []
for path in file_paths:
    all_chunks.extend(parse_html(path))

# Save structured chunks to JSON
with open("processed_chunks_with_links.json", "w") as f:
    json.dump(all_chunks, f, indent=4)
