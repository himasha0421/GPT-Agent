"""this module for user based google search query fetching"""

from langchain.utilities import GoogleSerperAPIWrapper
from utils import prompts


class Search:
    def __init__(self):
        print(f"Initializing Google Search Engine")
        self.search_api = GoogleSerperAPIWrapper()                           # initialize the google search wrapper     

    # define the prompt helpers
    @prompts(name="Search/Browse Google From User Input Text",
             description="useful when you want to get universal knowledge about everyting from a user input text and output relevant inforamtion "
                         "like:  provide me facts using web for this query, or is this query correct according to the universal knowledge? . "
                         "The input to this tool should be a string, representing the user input request used to browse the google. ")
    
    def inference(self, user_query):
        
        # return the google featched data
        return self.search_api.run(user_query)