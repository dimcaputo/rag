# rag
Training on RAG

This project uses uv as the environement manager.
To install uv:
~~~bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

Please download and install Ollama prior to running the scripts (https://ollama.com/download).

The embedding model will be pulled by the scripts.

To build the database:

~~~bash
$ uv run ingestion.py <path/to/folder/containing/pdfs>
~~~

To query the database:

~~~bash
$ uv run query.py <your query>
~~~

To test that the retriever is working properly: 
~~~bash
$ uv run test_retriever.py
~~~