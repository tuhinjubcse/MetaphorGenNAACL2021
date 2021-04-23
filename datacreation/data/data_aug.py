import csv

import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
mlm = RobertaForMaskedLM.from_pretrained("roberta-base")

lemmatizer = WordNetLemmatizer()

def loadData(f):
    f.readline()
    f_csv = csv.reader(f)
    data = []
    p = 0
    n = 0
    for line in tqdm(f_csv):
        verb = line[2]
        sentence = line[3].lower().split(" ")
        verb_posi = int(line[4])
        label = int(line[5])
        if label == 0:
            n += 1
            continue
        p += 1

        tokens = []
        new_verb_posi = 0
        for i in range(len(sentence)):
            word = " " + sentence[i] if i else sentence[i]
            token_tmp = tokenizer.tokenize(word)
            if i == verb_posi:
                new_verb_posi = len(tokens)
            tokens += token_tmp

        tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
        new_verb_posi += 1

        ids = tokenizer.convert_tokens_to_ids(tokens)

        data.append((ids, new_verb_posi, sentence, verb_posi))
    return data, p, n


with open("VUA_verb/VUA_formatted_train_noVAL.csv", "r") as f:
    train_dataset, train_positive, train_negative = loadData(f)
with open("VUA_verb/VUA_formatted_val.csv", "r") as f:
    valid_dataset, valid_positive, valid_negative = loadData(f)

print("data ready.")
print("train dataset: tot: %d, positive: %d, negative: %d" % (train_positive + train_negative, train_positive, train_negative))
print("valid dataset: tot: %d, positive: %d, negative: %d" % (valid_positive + valid_negative, valid_positive, valid_negative))


def augmentation(dataset):
    data = []
    for ids, new_verb_posi, sentence, verb_posi in tqdm(dataset):
        word_id = ids[new_verb_posi]
        ids[new_verb_posi] = tokenizer.mask_token_id
        score = mlm(torch.tensor(ids, dtype=torch.int64).view(1, -1))[0][0][new_verb_posi].detach()
        pred = torch.argsort(score, descending=True).detach().numpy().tolist()
        for each in pred:
            if each != word_id:
                verb= tokenizer._convert_id_to_token(each)
                if verb.startswith("Ġ"):
                    verb = verb[1:]
                sentence[verb_posi] = verb
                verb = lemmatizer.lemmatize(verb, "v")
                data.append((None, None, verb, " ".join(sentence), verb_posi, 0))
                break
    return data


def storeData(f, data):
    f_csv = csv.writer(f)
    f_csv.writerow(["text_idx", "sentence_idx", "verb", "sentence", "verb_idx", "label"])
    f_csv.writerows(data)


with open("VUA_verb/data_augmentation_train.csv", "w", encoding="utf-8") as f:
    storeData(f, augmentation(train_dataset))
with open("VUA_verb/data_augmentation_valid.csv", "w", encoding="utf-8") as f:
    storeData(f, augmentation(valid_dataset))
