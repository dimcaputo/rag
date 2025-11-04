
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_milvus import Milvus
import sys

if __name__ == "__main__":

    query = sys.argv[1]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "./milvus_example.db"},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search_with_score(query, k=10)

    response = retriever.batch(
        [
            query
        ],
    )


    for doc in sorted(response[0], key=lambda x:x[1], reverse=True):
        print("\n")
        print(f"Score = {doc[1]:.2f}")
        print(doc[0].page_content)