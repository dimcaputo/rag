
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
import os
from langchain_docling.loader import ExportType
import sys


if __name__ == "__main__":

    folder_to_check = sys.argv[1]

    filepaths = []
    for root, dirs, files in os.walk(folder_to_check):
        for file in files:
            filepaths.append(os.path.join(root, file))

    print("## LOADING FILES ######################################")
    loader = DoclingLoader(
        file_path=filepaths,
        export_type=ExportType.MARKDOWN
    )

    docs = loader.load()

    print("## SPLITTING FILES ####################################")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print("## LOADING OR CREATING DATABASE #######################")
    os.system("ollama pull nomic-embed-text")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "./milvus_example.db"},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )


    print("## INGESTING FILES ####################################")
    ids = vector_store.add_documents(documents=all_splits, nullable=True)