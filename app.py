import chainlit as cl
from phoenix.otel import register
import phoenix as ph
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.agents.middleware import dynamic_prompt, ModelRequest


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

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

        return system_message
    
    tools = []

    agent = create_agent(model, tools, middleware=[prompt_with_context])
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    # message_history = cl.user_session.get("message_history")

    agent = cl.user_session.get("agent")

    reply = await cl.Message(content="").send()

    for token, _ in agent.stream(
            {"messages": [{"role": "user", "content": message.content}]},
            stream_mode="messages",
        ):
            await reply.stream_token(token.content)


    await reply.update()
            

