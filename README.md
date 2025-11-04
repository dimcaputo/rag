# rag
Training on RAG

Please download and install Ollama prior to running the scripts (https://ollama.com/download).

The embedding model will be pulled by the scripts.

To build the database:

~~~bash
$ uv run ingestion.py <path/to/folder/containing/pdfs>
~~~

To query the database:

~~~bash
$ uv run query.py "your query"
~~~
