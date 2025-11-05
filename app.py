import chainlit as cl
from agent import Inquisita


inquisita = Inquisita()

@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    inquisita.query(message.content)
    # Send a response back to the user
    await cl.Message(
        content=f"{inquisita.reply}",
    ).send()