import csv
import json
import multiprocessing
import sys

import spacy
from nltk.stem import WordNetLemmatizer
from spacy_langdetect import LanguageDetector
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


# detect language
sentences = []
with open("gutenberg-poetry-v001.ndjson", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        line = json.loads(line)
        sentence = line["s"]
        doc = nlp(sentence)
        if doc._.language['language'] != "en":
            continue
        if doc._.language['score'] < 0.9:
            continue
        sentences.append(sentence)


# pos tag (parallel)

# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip and unzip
corenlp = StanfordCoreNLP("../data/corenlp/stanford-corenlp-full-2018-10-05")

def pos_tag(lines):
    lemmatizer = WordNetLemmatizer()

    data = []
    for line in lines:
        line = line.strip()
        pos = nlp.pos_tag(line)
        sentence = " ".join([each[0] for each in pos])
        for i, each in enumerate(pos):
            if each[1] in ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]:
                verb = lemmatizer.lemmatize(each[0].lower(), "v")
                data.append([sentence, verb, i])
    return data

num_processor = 8

tot = len(sentences)
sep = len(sentences) // num_processor
pool = multiprocessing.Pool(processes=num_processor)

result = []
for i in range(num_processor):
    data = pool.apply_async(pos_tag, (sentences[i*sep:((i+1)*sep if i != 59 else tot)],))
    result.append(data)
pool.close()
pool.join()
nlp.close()

dataset = []
for each in result:
    dataset.extend(each.get())

with open("poem_dataset.csv", "w") as f:
    f_csv = csv.writer(f)
    f_csv.writerows(dataset)
