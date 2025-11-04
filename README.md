# rag
Training on RAG

To build the database:

~~~bash
$ uv run ingestion.py path/to/folder/containing/pdfs
~~~

To query the database:

~~~bash
$ uv run query.py "query"
~~~
