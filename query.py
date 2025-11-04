
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_milvus import Milvus
import sys
import os
import requests
import ollama


req = requests.get("http://localhost:11434/api/tags").json()
model_is_nomic = [True for model in req['models'] if "nomic-embed-text" in model['name']]
if not any(model_is_nomic):
    os.system("ollama pull nomic-embed-text")
    
    
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./milvus_example.db"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search_with_score(query, k=10)

def get_results(query: str, retriever_func) -> list:
    results = {}
    response = retriever_func.batch(
        [
            query
        ],
    )
    for i, doc in enumerate(sorted(response[0], key=lambda x:x[1], reverse=True)):
        results[i]= {'score':round(doc[1], 2), 'chunk':doc[0].page_content}

    return results

if __name__ == "__main__":

    query = sys.argv[1]

    results = get_results(query, retriever)

    for i,item in results:
        print(f"Score = {item['score']}")
        print("Chunk " + "*"*50)
        print(item['chunk'])
        print('*'*50)
        print('\n')
