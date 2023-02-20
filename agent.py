# we are building a langchain conversational agent
from langchain import OpenAI
from langchain.agents import initialize_agent , Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit   as st


# define a function to calculate the nth fibonacci number
def fib(n) :
    if n <= 1:
        return n
    else:
        return ( fib( n-1 ) + fib( n-2 ) )

# this function sorts the input string alphabetically
def sort_string( string ):
    return ''.join( sorted( string ) )


# defien a function to trun the word into an encrypted word
def encrypt(word):
    encrypted_word = ''
    for letter in word:
        encrypted_word += chr( ord( letter ) + 1 )
    return encrypted_word


# define a function to encrypted word into decrypt mode
def decrypt(word):
    decrypted_word = ''
    for letter in word :
        decrypted_word += chr( ord( letter ) -1 )
    return decrypted_word


tools = [

    Tool(
        name = 'Fibonacci' ,
        func= lambda n : str( fib(int(n)) ) ,
        description="use when you want to calculate the nth fibonacci number",
        #return_direct=True
    ) ,
    Tool(
        name='Sort String' ,
        func=lambda string : sort_string(string) ,
        description="use when you want to sort  a string alphabetically",
        #return_direct=True
    ),
    Tool(
        name='Encrypt',
        func=lambda word : encrypt(word),
        description="use when you want to encrypt a word",
        #return_direct=True
    ),
    Tool(
        name="Decrypt",
        func= lambda word : decrypt(word) ,
        description="use when you want to decrypt a word",
        #return_direct=True
    )


]


from pyannote.audio import Pipeline
from dotenv import load_dotenv
load_dotenv()
import os

api_key =  os.environ["OPENAI_API_KEY"]


# define the chat memeory buffer
memory = ConversationBufferMemory( memory_key='chat_history' ) # --> try other memeory types as well
# define the openai LLM ]
llm =  OpenAI( temperature= 0.1 , verbose=True )
agent_chain =  initialize_agent(  tools=tools ,
                                   llm= llm ,
                                   agent= "conversational-react-description" ,
                                   memory  = memory ,
                                   verbose = True   # verbose handle the agents though process  
 )


# initialize the steamlit app
st.header( ":blue[Langchain Chatbot with agent/tools and memeory] :sunglasses:" ) # short term memory no embedd vector store
user_input =  st.text_input("You: ")
# initialize the memory buffer
if "memeory" not in st.session_state :
    st.session_state['memory'] = ""

# add stremalit button
if st.button("Submit"):

    # execute the langchain agent
    st.markdown( agent_chain.run( input =  user_input ) )
    # print the memeory buffer
    # add the conversation history to memeory buffer
    st.session_state['memory'] += memory.buffer
    print( st.session_state['memory'] )





