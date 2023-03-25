import re

import streamlit as st
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper

# load the enviroment variables
load_dotenv()

def check_balance(word:str) ->str:
    """this function is to query the backend system to check balance"""
    answer = word
    msisdn = [str(s) for s in re.findall(r"\b\d+\b", answer)]

    if msisdn == []:
        return "Please Input the mobile number to check the account balance ..."
    else:
        return "Number : {} \nCurrent Balance is : {}".format(msisdn[0], "Rs.10")


# initialize the google search wrapper
SEARCH = GoogleSerperAPIWrapper()

# define the tool for the agent for querying google
TOOLS = [
    Tool(
        name="Search",
        func=SEARCH.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True,
    ),
    Tool(
        name="Check Account Balance",
        func=lambda number: check_balance(number),
        description="useful for when you need to get the account balance for the mobile number",
        return_direct=True,
    ),
]




####################################################  define the langchain agent #########################################################
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
SUFFIX = """When answering, you MUST speak in the following language: {language}.

            Question: {input}
            {agent_scratchpad}"""  # we currently require an agent_scratchpad input variable to put notes on previous actions and observations

# create a prompt template matching with the zeroshot agent
PROMPT = ZeroShotAgent.create_prompt(
    tools=TOOLS,
    prefix= PREFIX ,
    suffix= SUFFIX ,
    input_variables=["input", "language", "agent_scratchpad"],
)

# define the multi lingual llmchain
MULTI_LINGUAL_LLM = LLMChain(llm=OpenAI(temperature=0),
                             prompt= PROMPT )


# define the chat memeory buffer
MEMORY = ConversationBufferMemory(memory_key="chat_history")  # --> try other memeory types as well

# define the zeroshot agent
MULTI_LINGUAL_AGENT = ZeroShotAgent(
    llm_chain=MULTI_LINGUAL_LLM ,
    tools=TOOLS,
    agent="conversational-react-description",
    memory=MEMORY,
)

# define the agent executor
MULTI_LINGUAL_EXECUTOR= AgentExecutor.from_agent_and_tools(
    agent=MULTI_LINGUAL_AGENT, tools=TOOLS, verbose=True
)

################################## define the streamlit app #############################

# initialize the steamlit app
st.header(
    ":blue[ ChatGPT Agent ] :sunglasses:"
)  # short term memory no embedd vector store
USER_INPUT= st.text_input("Customer : ")
# initialize the memory buffer
if "memory" not in st.session_state:
    st.session_state["memory"] = ""

# add stremalit button
if st.button("Submit"):
    # execute the langchain agent
    st.markdown(MULTI_LINGUAL_EXECUTOR.run(input=USER_INPUT, language="Enaglish"))
    # print the memeory buffer
    # add the conversation history to memeory buffer
    st.session_state["memory"] += MEMORY.buffer
    print(st.session_state["memory"])
