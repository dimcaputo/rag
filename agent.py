
from langchain_ollama import OllamaEmbeddings
import os
import requests
from langchain_chroma import Chroma
import sys
from ollama import chat, AsyncClient
import json
from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes
import phoenix as px


class Inquisita:
    def __init__(self):
        session = px.launch_app() 
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


    def query(self, query):
        tracer_provider = register(project_name="Inquisita", auto_instrument=True)
        tracer = tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("chat_interaction") as span:
            # Set the user query as a span attribute
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)
            
            context, retrieved_docs = self.retrieve_context(query)
            span.set_attribute(SpanAttributes.METADATA, json.dumps(context, indent=4))
            span.set_attribute(SpanAttributes.RETRIEVAL_DOCUMENTS, str(retrieved_docs))


            messages = [{'role': 'user', 'content': f'{query}. Please take into account the following information: {context}'}]
            span.set_attribute(SpanAttributes.LLM_PROMPTS, messages[0]['content'])


            self.stream = chat(model='gemma3:4b', messages=messages, stream=True)
            self.reply = ""
            for chunk in self.stream:
                self.reply += chunk['message']['content']
                print(chunk['message']['content'], end='', flush=True)
            
            # Set the model reply as a span attribute
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, self.reply)
            

