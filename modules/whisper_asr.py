""" this module responsible to convert audio into text output
    use state of the art whisper model
"""
# initiaizations
import os

import numpy as np
import torch
import whisper
from dotenv import load_dotenv
from utils import write_vtt

load_dotenv()
import gc

#torch.multiprocessing.set_start_method('spawn')

class openai_whisper():
    def __init__(self) -> None:
        """class init"""

        # define the underline device architecture and clean memeory
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        print("device set :", self.device)
        torch.cuda.empty_cache()
        gc.collect()

        # load the whisper model
        self.model_whisper =  whisper.load_model('medium', device = self.device)

        # print model stats
        print(
            f"Model is {'multilingual' if self.model_whisper.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in self.model_whisper.parameters()):,} parameters."
        )


    # step 2. transcribe the audio
    def transcribe(self, filename, vtt_filepath):

        # load the audio file
        audio = whisper.load_audio(filename)
        # pad or trim the audio to match model input size
        audio = whisper.pad_or_trim(audio)
        # extract the mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # first detect the lanaguge
        _ , probs = self.model_whisper.detect_language(mel)
        print(f"Detected Language : { max( probs, key=probs.get ) } ")

        # define the  parameters
        beam_size=5
        best_of=5
        temperature=0.0
        detected_lang = max(probs, key=probs.get) 
        
        # define transcribe params
        decode_options = dict(language= detected_lang, best_of=best_of, beam_size=beam_size, temperature=temperature)
        transcribe_options = dict(task="transcribe", **decode_options)
        
        # transcribe the audio
        transcribe = self.model_whisper.transcribe(audio= filename, **transcribe_options) 

        # save VTT
        out_path =  f"data/transcribe_results/{vtt_filepath}"
        with open(  out_path , "w", encoding="utf-8") as vtt:
            write_vtt( transcribe["segments"], file=vtt)

        print("Transcribe Successfull  !!!!!!!!!!!")

        return out_path 






