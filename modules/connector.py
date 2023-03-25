"""this module responsible to create the connection between different
    modules like whisper asr , langchain chains , langain agents , intent recognition
"""

"""this module is the main app execute the gradio app with QnA bot"""
import datetime
import os

import gradio as gr
import langchain
import weaviate
from agent import ChatAgent
from chain import get_qna_chain
from dotenv import load_dotenv
from intent_recognition import Intent_Agent
from langchain.chains.base import Chain
from langchain.vectorstores import Weaviate

# load the enviroment variables from the .env file
load_dotenv()
# load the vector database url
WEAVIATE_URL = os.environ["WEAVIATE_URL"]

class Connector:
    """this class connect different components whisper asr , langain agent/chain"""
    def __init__(self , vectorstore, openai_key) -> None:
        self.vectorstore = vectorstore
        # init the agent and lang chain object
        self.backend_executor = self.init_agent()
        self.qna_chain = self.init_qa_chain(openai_key)
        self.intent_classifier =  Intent_Agent()

    def init_qa_chain(self, api_key) -> Chain:
        """init vector store and init the qna chain"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            # init the qna langchain
            qa_chain = get_qna_chain(self.vectorstore)
            return qa_chain
        
    def init_agent(self ):
        """initialize the langchain backend query agent"""
        # initialize the chat agent
        backend_agent = ChatAgent()
        # initialize the executor
        backend_executor = backend_agent.init_agent()
        return backend_executor
    
    def ask(self, input_query, history ):
        """ this is the main workflow function
            apply inteinnt classification and indentify the intention
            based on the intent apply backend executor or qna agent
        """
        # step1. apply intent classification
        intent = self.intent_classifier.classify(input_text=input_query)

        #step2. if intent is to check backend then forward to agent executor
        if(intent in ('info_system', 'general')):
            result = self.backend_executor.run(input=input_query, language="English")
        # otherwise direct to langchain qna chain
        else:
            # call the agent with question and chat history
            output = self.qna_chain({"question":input_query, "chat_history":history})
            result = output["answer"]

        return result




