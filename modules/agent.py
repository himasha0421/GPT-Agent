"""this module responsible for check backend systems or for web scraping"""
from typing import List

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
from utils import check_balance

# load the enviroment variables
load_dotenv()

# initialize the agent
class ChatAgent :
    """this class responsible to act as backend query system"""

    llm_model = OpenAI(temperature=0)                               # define the openai llm
    memory = ConversationBufferMemory(memory_key="chat_history")    # define chathistory buffer memeory
    search_api = GoogleSerperAPIWrapper()                           # initialize the google search wrapper                    

    def __init__(self) -> None:
        pass

    def init_tools(self) -> List[Tool]:
        """this method responsible to initialize serveral tools google api/balance check"""

        tools = [
            Tool(
                name="Search",
                func= self.search_api.run,
                description="useful for when you need to answer questions about current events",
                return_direct=False,
            ),
            Tool(
                name="Check Account Balance",
                func=lambda number: check_balance(number),
                description="useful for when you need to get the account balance for the mobile number",
                return_direct=True,
            ),
        ]
        return tools
    
    def init_prompt(self):
        """this method responsible to initialize the agent prompt"""

        prefix = """Answer the following questions as best you can. You have access to the following tools:"""
        suffix = """When answering, you MUST provide answer in more human readable way like call center agent , provide more facts as possible don't shorten
                    the answer too much and speak in the following language: {language}.

                    Question: {input}
                    {agent_scratchpad}"""  # we currently require an agent_scratchpad input variable to put notes on previous actions and observations
        return prefix , suffix
    
    def init_agent(self) -> AgentExecutor:
        """this method responsible to initialize the chat agent for backend system query"""

        # step 1. get tools
        tools = self.init_tools()

        # step 2. init prompts
        prefix, suffix = self.init_prompt()

        # create a prompt template matching with the zeroshot agent
        prompt = ZeroShotAgent.create_prompt(
            tools= tools,
            prefix= prefix,
            suffix= suffix,
            input_variables=["input", "language", "agent_scratchpad"],
        )

        # define the multi lingual llmchain
        multi_lingual_llm = LLMChain(llm= self.llm_model,
                                    prompt= prompt )

        # define the zeroshot agent
        multi_lingual_agent = ZeroShotAgent(
            llm_chain=multi_lingual_llm,
            tools=tools,
            agent="conversational-react-description",
            memory=self.memory,
        )

        # define the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent= multi_lingual_agent,
            tools=tools, 
            verbose=True
        )

        return agent_executor

"""
Insturctions to use the agent class

agent_class =  ChatAgent()
agent_executor =  agent_class.init_agent()
check agent results using --> agent_executor.run(input=USER_INPUT, language="Enaglish"))
"""