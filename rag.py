from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_milvus import Milvus


file_path = "paper.pdf"



loader = DoclingLoader(file_path=file_path)

docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./milvus_example.db"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)



ids = vector_store.add_documents(documents=all_splits)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search_with_score(query, k=10)


response = retriever.batch(
    [
        ""
    ],
)

print(response[0])