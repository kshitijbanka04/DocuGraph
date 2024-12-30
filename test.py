import os

persist_directory = "/Users/kshitijbanka/DocuGraph/vector_db"

if os.path.exists(persist_directory):
    print("Vector DB directory exists!")
else:
    print("Vector DB directory does not exist.")