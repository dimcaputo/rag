

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama

import os

import sys
import requests


from langchain_community.document_loaders.parsers import TesseractBlobParser

from langchain_community.document_loaders import PDFMinerLoader


import dotenv
from langchain_chroma import Chroma


if __name__ == "__main__":

    dotenv.load_dotenv()

    folder_to_check = sys.argv[1]

    if not os.path.exists(folder_to_check):
        raise FileNotFoundError('Please input a valid path.')

    filepaths = []
    for root, dirs, files in os.walk(folder_to_check):
        for file in files:
            filepaths.append(os.path.join(root, file))

    print(f"\n{len(filepaths)} pdf files will be loaded.\n")

    
    for filepath in filepaths:
        print("\n## LOADING FILES ######################################\n")
        loader = PDFMinerLoader(
            filepath,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=TesseractBlobParser(),
        )

        docs = loader.load()
        print(docs[0].page_content)

        print(f"\nThe {len(filepaths)} pdf files were loaded and gave {len(docs)} langchain documents.\n")

        print("\n## SPLITTING FILES ####################################\n")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # chunk size (characters)
            chunk_overlap=5,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        
        print(f"\n{len(all_splits)} splits were created\n")

        if os.path.exists("./chorma_langchain_db"):
            print("\n## LOADING DATABASE ################################\n")
        else:
            print("\n## CREATING DATABASE ###############################\n")
        
        
        req = requests.get("http://localhost:11434/api/tags").json()
        model_is_nomic = [True for model in req['models'] if "nomic-embed-text" in model['name']]
        if not any(model_is_nomic):
            print("Pulling ollama embedding model...")
            os.system("ollama pull nomic-embed-text")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if os.path.exists("./chroma_langchain_db"):
            print("\nDatabase at ./chroma_langchain_db loaded\n")
        else:
            print("\nDatabase at ./chroma_langchain_db created\n")

        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
        


        print("\n## INGESTING FILES ####################################\n")
        ids = vector_store.add_documents(documents=all_splits)
        