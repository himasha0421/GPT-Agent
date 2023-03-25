"""base module to get the intent type using cohere embedding model"""
import os
from typing import List

import cohere
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# load the enviroment variables from the .env file
load_dotenv()

# define the api key
API_KEY = os.environ["COHERE_API_KEY"]
# initiate the cohere client
COHERE_CLIENT = cohere.Client(API_KEY)

class Intent_Agent:
    """this class responsible to classify the intent into categories using kmeans model"""

    # test the model on custom query and get the cluster assignment
    INTENT_MAP = {
        0 : 'info_system', 
        1 : 'langchain',
        2 : 'general',
        3 : 'complain'
    }

    def __init__(self) -> None:
        # step 1--> load the kmeans model
        self.KMEANS_MODEL = self.load_sklearn_model("models/kmean_model.joblib")

    # Get text embeddings
    def get_embeddings(self, texts:List[str], model='large')->List[List[float]] :
        """apply cohere embedding model on text input"""
        output = COHERE_CLIENT.embed(
                        model=model,
                        texts=texts,
                        truncate='RIGHT'
                        )
        # get the embedding output
        return output.embeddings

    # load the model from the path
    def load_sklearn_model(self, file_path:str ):
        """load the sklearn trained kmeans model from joblib object"""
        model =  joblib.load( file_path )
        return model
    
    def classify(self, input_text:str) ->str:
        """convert the text into embedding and apply kmeans classifier"""
        #step1 --> embed the input text
        INPUT_EMBED =  self.get_embeddings( [input_text] )[0]
        # get the cluster assignment
        TEXT_CLASS = self.KMEANS_MODEL.predict([ INPUT_EMBED ]).tolist()[0]
        # print the final results
        print(f"Input Text : {input_text} Mapped Intent Class : {self.INTENT_MAP[TEXT_CLASS]} ")

        return self.INTENT_MAP[TEXT_CLASS]

