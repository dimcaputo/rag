# My first RAG

This project uses uv as the environement manager.
To install uv:
~~~bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

Please get a Google API key and save it in a .env file:
~~~bash
$ touch .env
$ echo GOOGLE_API_KEY=your_key >> .env
~~~

Please download and install Ollama prior to running the scripts (https://ollama.com/download).
The embedding model will be pulled by the scripts.

To build the database:

~~~bash
$ uv run ingestion.py <path/to/folder/containing/pdfs>
~~~

To launch the app:

~~~bash
$ uv run chainlit run app.py -w
~~~