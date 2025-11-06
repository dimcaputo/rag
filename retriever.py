
from langchain_ollama import OllamaEmbeddings
import os
import requests
from langchain_chroma import Chroma


class Inquisita:
    def __init__(self):
        
        self.req = requests.get("http://localhost:11434/api/tags").json()
        model_is_nomic = [True for model in self.req['models'] if "nomic-embed-text" in model['name']]
        if not any(model_is_nomic):
            os.system("ollama pull nomic-embed-text")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db"
        )

    def retrieve_context(self, query:str):
        """Retrieve information to help answer a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
            

