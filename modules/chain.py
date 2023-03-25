"""this module responsible to create the langchain custom chain for knowldege base
question answering bot
"""

import json
import os
import pathlib
from typing import Dict, List, Tuple

import weaviate
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS, Weaviate
from pydantic import BaseModel

# load the enviroment variables from the .env file
load_dotenv()

class CustomChain(Chain, BaseModel):
    """class initialize the custom qna chain"""
    vstore: Weaviate
    chain: BaseCombineDocumentsChain
    key_word_extractor: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """call the chain with input"""

        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        print( "Question : ", question, "\nChat History : ", chat_history_str)  # to check the question and chat history
        if chat_history_str:
            # rephrase the question based on input and history context
            new_question = self.key_word_extractor.run(
                question=question, 
                chat_history=chat_history_str
            )
        else:
            new_question = question

        print( "New Question :" , new_question)
        # search new docs related to new question using semantic search --> ouput top 4
        docs = self.vstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        # apply the chain with docs combined and get the final answer
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        return {"answer": answer}


def get_qna_chain(vectorstore) ->Chain:
    """this function responsible to intialize the chain with prompts/llms"""

    # step 1. initialize the vector database
    weaviate_url = os.environ["WEAVIATE_URL"]
    client = weaviate.Client(
        url=weaviate_url,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    # set the example template for question rephrasing
    _eg_template = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: {answer}"""

    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["chat_history", "question", "answer"],
    )

    _prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. You should assume that the question is related to LangChain."""
    _suffix = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    eg_store = Weaviate(
        client,
        "Rephrase",
        "content",
        attributes=["question", "answer", "chat_history"],
    )
    # set the example selector to be sementic selector --> ouput top 4
    example_selector = SemanticSimilarityExampleSelector(vectorstore=eg_store, k=4)
    # define the fewshot template
    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        example_selector=example_selector,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )
    # define the llm to use with question rephrasing
    llm = OpenAI(temperature=0, 
                 model_name="text-davinci-003",
                 max_tokens=-1)
    # define the keyword extractor LLM Chain
    key_word_extractor = LLMChain(llm=llm, prompt=prompt)

    # vector store example template --> content , source
    example_prompt = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )
    # define the template to answer the question based on the given context
    template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
        You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
        If the question includes a request for code, provide a code block directly from the documentation.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:"""
    chain_prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    # initialize the qna chain
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1),
        chain_type="stuff",
        prompt=chain_prompt,
        document_prompt=example_prompt,
    )
    return CustomChain(
        chain=doc_chain, vstore=vectorstore, key_word_extractor=key_word_extractor
    )


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    """this function responsible to get the chat history into final string"""
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer
