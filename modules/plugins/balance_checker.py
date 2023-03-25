"""this plugin mainly for check customer account balance"""
import re

import numpy as np
from utils import prompts


class BalanceChecker:
    def __init__(self):
        print(f"Initializing Balance Checker Module")
        self.neg_out = "Please Input the mobile number to check the account balance ..."
        self.pos_out = "Number : {} \nCurrent Balance is : Rs. {}"

    # define the prompt helpers
    @prompts(name="Check Customer Account Credit Balance From User Input Text",
             description="useful when you want to check account balance from a user input text and output account balance value. "
                         "like:  check credit balance for 123456, or please check balance for provided number 0764523689. "
                         "The input to this tool should be a number without any strings inside, representing the mobile number used to check the balance. ")
    
    def inference(self, mobile_number):
        #verify the number
        msisdn = [str(s) for s in re.findall(r"\b\d+\b", mobile_number)]
        balance = np.random.choice( np.arange(500) , size=1 )

        if msisdn==[]:
            return self.neg_out
        else:
            return self.pos_out.format(msisdn, balance)