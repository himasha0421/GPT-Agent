"""this module is the main app execute the gradio app with QnA bot"""
import datetime
import os

import gradio as gr
import langchain
import weaviate
from chain import get_qna_chain
from connector import Connector
from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.vectorstores import Weaviate

# load the enviroment variables from the .env file
load_dotenv()
# load the vector database url
WEAVIATE_URL = os.environ["WEAVIATE_URL"]


def get_weaviate_store() -> Weaviate:
    """initialize the vector database clint using openai & weaviate key"""
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    return Weaviate(client, "Paragraph", "content", attributes=["source"])

def set_openai_api_key(api_key, agent) -> Chain:
    """init vector store and init the qna chain"""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        # init the vector db
        vectorstore = get_weaviate_store()
        # init the connector object
        agent = Connector(vectorstore=vectorstore, openai_key=api_key)

        return agent

def chat(inp, history, agent):
    """main chat function"""
    history = history or []

    # if chat agent not initialized the show warning
    if agent is None:
        history.append((inp, "Please paste your OpenAI key to use"))
        # output history twise because of one for gradio chatbot object other to track chat state
        return history, history
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)

    # ----- execute the workflow ------------

    # call the agent with question and chat history
    answer = agent.ask(inp, history)
    # add the text pair , question and answer into history
    history.append((inp, answer))
    # output history twise because of one for gradio chatbot object other to track chat state
    return history, history

# -------------------- init gradio application ------------------------------

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain AI</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What are agents?",
            "How do I summarize a long document?",
            "What types of memory exist?",
        ],
        inputs=message,
    )

    gr.HTML(
        """
    This simple application is an implementation of ChatGPT but over an external dataset (in this case, the LangChain documentation)."""
    )
    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )
    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox, agent_state],
        outputs=[agent_state],
    )

block.launch(debug=True , share=True)
