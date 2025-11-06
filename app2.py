from openai import AsyncOpenAI
import chainlit as cl
from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes
import phoenix as ph
from retriever import Inquisita
import json

session = ph.launch_app()
tracer_provider = register(project_name="LLM App", auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)

retriever = Inquisita()

# Point to Ollama instead of OpenAI
client = AsyncOpenAI(
    api_key="ollama",  # dummy key, Ollama doesn’t require auth
    base_url="http://localhost:11434/v1"
)



settings = {
      # or any model you pulled with Ollama
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

@cl.on_message
async def main(message: cl.Message):
    with tracer.start_as_current_span("chat_interaction") as span:
        message_history = cl.user_session.get("message_history")

        span.set_attribute(SpanAttributes.INPUT_VALUE, message.content)

        info, docs = retriever.retrieve_context(message.content)

        prompt = {'role': 'user', 'content': f'{message.content}. Please take into account the following information: {info}'}

        span.set_attribute(SpanAttributes.LLM_PROMPTS, json.dumps(prompt, indent=4))

        message_history.append(prompt)

        msg = cl.Message(content="")

        stream = await client.chat.completions.create(
            model = "gemma3:4b", 
            messages=message_history, 
            stream=True, 
            **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, msg.content)
