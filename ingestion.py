
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
import os
from langchain_docling.loader import ExportType
import sys
import requests


if __name__ == "__main__":

    folder_to_check = sys.argv[1]

    if not os.path.exists(folder_to_check):
        raise FileNotFoundError('Please input a valid path.')

    filepaths = []
    for root, dirs, files in os.walk(folder_to_check):
        for file in files:
            filepaths.append(os.path.join(root, file))

    print(f"\n{len(filepaths)} pdf files will be loaded.\n")

    print("\n## LOADING FILES ######################################\n")
    loader = DoclingLoader(
        file_path=filepaths,
        export_type=ExportType.MARKDOWN
    )
    
    docs = loader.load()
    print(f"\nThe {len(filepaths)} pdf files were loaded and gave {len(docs)} langchain documents.\n")

    print("\n## SPLITTING FILES ####################################\n")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"\n{len(all_splits)} splits were created\n")

    if os.path.exists("./milvus_example.db"):
        print("\n## LOADING DATABASE ################################\n")
    else:
        print("\n## CREATING DATABASE ###############################\n")
    
    
    req = requests.get("http://localhost:11434/api/tags").json()
    model_is_nomic = [True for model in req['models'] if "nomic-embed-text" in model['name']]
    if not any(model_is_nomic):
        print("Pulling ollama embedding model...")
        os.system("ollama pull nomic-embed-text")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "./milvus_example.db"},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    
    if os.path.exists("./milvus_example.db"):
        print("\nDatabase at ./milvus_example.db loaded\n")
    else:
        print("\nDatabase at ./milvus_example.db created\n")

    print("\n## INGESTING FILES ####################################\n")
    ids = vector_store.add_documents(documents=all_splits)
    print("\nThe files were ingested into the database.\n")