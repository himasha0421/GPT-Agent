import os

import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory


# define a function to calculate the nth fibonacci number
def fib(number: int) -> int:
    """this is the function to calculate fibonacci number"""
    if number <= 1 :
        return number
    else:
        return fib(number - 1) + fib(number - 2)


# this function sorts the input string alphabetically
def sort_string(string: str) -> str:
    """this funct6ion sort a string in alphebetical order"""
    return "".join(sorted(string))

# defien a function to trun the word into an encrypted word
def encrypt(word: str) -> str:
    """this function encrypt a word using alphebetical order index +1"""
    encrypted_word = ""
    for letter in word:
        encrypted_word += chr(ord(letter) + 1)
    return encrypted_word

# define a function to encrypted word into decrypt mode
def decrypt(word: str) -> str:
    """this function decurypt a word given encrypted using above method"""
    decrypted_word = ""
    for letter in word:
        decrypted_word += chr(ord(letter) - 1)
    return decrypted_word


TOOLS = [
    Tool(
        name="Fibonacci",
        func=lambda n: str(fib(int(n))),
        description="use when you want to calculate the nth fibonacci number",
        # return_direct=True
    ),
    Tool(
        name="Sort String",
        func=lambda string: sort_string(string),
        description="use when you want to sort  a string alphabetically",
        # return_direct=True
    ),
    Tool(
        name="Encrypt",
        func=lambda word: encrypt(word),
        description="use when you want to encrypt a word",
        # return_direct=True
    ),
    Tool(
        name="Decrypt",
        func=lambda word: decrypt(word),
        description="use when you want to decrypt a word",
        # return_direct=True
    ),
]


load_dotenv()


API_KEY = os.environ["OPENAI_API_KEY"]


# define the chat memeory buffer
MEMORY = ConversationBufferMemory(
    memory_key="chat_history"
)  # --> try other memeory types as well

# define the openai LLM
# type: ignore
LLM = OpenAI(temperature=0.1, verbose=True)

AGENT_CHAIN = initialize_agent(
    tools=TOOLS,
    llm=LLM,
    agent="conversational-react-description",
    memory=MEMORY,
    verbose=True,  # verbose handle the agents though process
)


# initialize the steamlit app
st.header(
    ":blue[Langchain Chatbot with agent/tools and memeory] :sunglasses:"
)  # short term memory no embedd vector store
USER_INPUT = st.text_input("You: ")
# initialize the memory buffer
if "memeory" not in st.session_state:
    st.session_state["memory"] = ""

# add stremalit button
if st.button("Submit"):
    # execute the langchain agent
    st.markdown(AGENT_CHAIN.run(input=USER_INPUT))
    # print the memeory buffer
    # add the conversation history to memeory buffer
    st.session_state["memory"] += MEMORY.buffer
    print(st.session_state["memory"])
