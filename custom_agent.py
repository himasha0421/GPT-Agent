from pyannote.audio import Pipeline
from dotenv import load_dotenv
load_dotenv()
import os

# import the la
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit   as st


# add a custom function to query the balance
import re

def check_balance(word):
    answer= word
    msisdn = [ str(s) for s in re.findall(r'\b\d+\b',  answer ) ]

    if( msisdn == [] ):
        return "Please Input the mobile number to check the account balance ..."
    else:
        return "Number : {} \nCurrent Balance is : {}".format( msisdn[0] , 'Rs.10' )
    

# initialize the google search wrapper
search = GoogleSerperAPIWrapper()

# define the tool for the agent for querying google
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True
    ),
    Tool(
        name = "Check Account Balance" ,
        func =  lambda number : check_balance( number ) ,
        description = "useful for when you need to get the account balance for the mobile number",
        return_direct=True
        
    )
]


####################################################  define the langchain agent #########################################################
prefix = """Answer the following questions as best you can. You have access to the following tools:"""
suffix = """When answering, you MUST speak in the following language: {language}.

Question: {input}
{agent_scratchpad}""" # we currently require an agent_scratchpad input variable to put notes on previous actions and observations

# create a prompt template matching with the zeroshot agent
prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "language", "agent_scratchpad"]
)

# define the multi lingual llmchain
multi_lingual_llm = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)


# define the chat memeory buffer
memory = ConversationBufferMemory( memory_key='chat_history' ) # --> try other memeory types as well

#define the zeroshot agent
multi_lingual_agent = ZeroShotAgent(llm_chain= multi_lingual_llm ,
                                    tools=tools ,
                                    agent= "conversational-react-description" ,
                                    memory  = memory ,
                                    )

# define the agent executor
multi_lingual_executor = AgentExecutor.from_agent_and_tools(agent= multi_lingual_agent  ,
                                                            tools=tools ,
                                                            verbose=True
                                                            )


########################################################## define the streamlit app ##################################################


# initialize the steamlit app
st.header( ":blue[ ChatGPT Agent ] :sunglasses:" ) # short term memory no embedd vector store
user_input =  st.text_input("Customer : ")
# initialize the memory buffer
if "memory" not in st.session_state :
    st.session_state['memory'] = ""

# add stremalit button
if st.button("Submit"):

    # execute the langchain agent
    st.markdown( multi_lingual_executor.run( input =  user_input  ,
                                             language = "Enaglish"
                                             ) )
    # print the memeory buffer
    # add the conversation history to memeory buffer
    st.session_state['memory'] += memory.buffer
    print( st.session_state['memory'] )