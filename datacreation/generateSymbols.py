import nltk
import yaml
import string
import json
from loadconfig import loadConfig
import os
import sys
import ast
from urllib.parse import quote
nltk.download('maxent_ne_chunker')
nltk.download('words')

sys.path.append(os.getcwd()+'/comet-commonsense')


def getCommonSense(utterance):
        os.system('python comet-commonsense/scripts/generate/generate_conceptnet_arbitrary.py --model_file comet-commonsense/pretrained_models/conceptnet_pretrained_model.pickle --input "'+utterance+'" --output_file output.json --device 0 --sampling_algorithm beam-5')
        output = json.load(open('output.json', "r"))
        return output[0]['SymbolOf']['beams']


def retrieveCommonSense(utterance):
        modified_utterance = utterance
        print(utterance)
        return getCommonSense(modified_utterance)


print(retrieveCommonSense(sys.argv[1]))



#Run as python generateSymbols "The flower danced in the garden"
#you can edit the number of beams you want to generate
