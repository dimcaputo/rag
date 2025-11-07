import chainlit as cl
from phoenix.otel import register
import phoenix as ph
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


session = ph.launch_app()
tracer_provider = register(project_name="LLM App", auto_instrument=True)

@cl.on_chat_start
def start_chat():

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(),
        "retriever",
        "Search and return information about Dimitri.",
    )

    response_model = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0)

    def generate_query_or_respond(state: MessagesState):
        response = (
            response_model
            .bind_tools([retriever_tool]).invoke(state["messages"])  
        )
        return {"messages": [response]}
    
    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Context: {context}"
    )

    def generate_answer(state: MessagesState):
        """Generate an answer."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    # workflow.add_conditional_edges(
    #     "retrieve",
    #     # Assess agent decision
    #     grade_documents,
    # )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()

    # prompt = (
    #         "You have access to a tool that retrieves context from the CV of Dimitri Caputo. "
    #         "Use it to inform your answers, but always provide a complete and natural language reply to the user."
    #     )

    # agent = create_agent(model, tools, system_prompt=prompt)
    cl.user_session.set("graph", graph)

@cl.on_message
async def main(message: cl.Message):
    # message_history = cl.user_session.get("message_history")

    graph = cl.user_session.get("graph")

    reply = cl.Message(content="")

    for token, _ in graph.stream(
            {"messages": [{"role": "user", "content": message.content}]},
            stream_mode="messages",
        ):
            await reply.stream_token(token.content)


    await reply.send()
            

