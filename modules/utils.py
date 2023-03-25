"""this module constits of all the supportive util functions"""

import re
from typing import Iterator, TextIO


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

def check_balance(word:str) ->str:
    """this function is to query the backend system to check balance"""
    answer = word
    msisdn = [str(s) for s in re.findall(r"\b\d+\b", answer)]

    if msisdn == []:
        return "Please Input the mobile number to check the account balance ..."
    else:
        return f"Number : {msisdn[0]} \nCurrent Balance is : {'Rs.10'}"
    

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    """this function format the timestep into hours:minutes:seconds:milliseconds """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def write_vtt(transcript: Iterator[dict], file: TextIO):
    """this function save the whisper transcript output into webvtt fileformat"""
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )