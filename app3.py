import chainlit as cl
from phoenix.otel import register
import phoenix as ph
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool


session = ph.launch_app()
tracer_provider = register(project_name="LLM App", auto_instrument=True)

@cl.on_chat_start
def start_chat():
    # cl.user_session.set(
    #     "message_history",
    #     [{"role": "system", "content": "You are a helpful assistant."}],
    # )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    tools = [retrieve_context]

    prompt = (
            "You have access to a tool that retrieves context from the CV of Dimitri Caputo. "
            "Use it to inform your answers, but always provide a complete and natural language reply to the user."
        )

    agent = create_agent(model, tools, system_prompt=prompt)
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    # message_history = cl.user_session.get("message_history")

    agent = cl.user_session.get("agent")

    reply = cl.Message(content="")

    for token, _ in agent.stream(
            {"messages": [{"role": "user", "content": message.content}]},
            stream_mode="messages",
        ):
            await reply.stream_token(token.content)


    await reply.send()
            

