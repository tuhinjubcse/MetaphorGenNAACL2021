import argparse
import csv
import json
import os
import random
import time
from string import punctuation

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import nltk
from nltk.stem import WordNetLemmatizer
from fairseq.models.roberta import RobertaModel

from .model import BertForMD

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str)
parser.add_argument("--model_file", type=str)

# parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--gpu", type=str, default="0")

args = parser.parse_args()

# if args.seed:
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

model_name = "bert-base-cased"

tokenizer = BertTokenizer.from_pretrained(model_name)
lemmatizer = WordNetLemmatizer()


def loadData(f):
    data = []
    for line in f:
        line = line.strip()
        words = line.split(" ")
        pos = nltk.pos_tag(words)
        for i, each in enumerate(pos):
            if each[1] in ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]:
                verb = lemmatizer.lemmatize(each[0].lower(), "v")
                if verb in ["be", "have", "do"]:
                    continue
                data.append([line, words.copy(), verb, i])
    return data


def mlm_fn(model, mlm, data, f_ans, top=200):
    results = []
    for sentence, words, verb, verb_posi in tqdm(data):
        words[verb_posi] = "<mask>"
        masked_sentence = " ".join(words)
        mlm_result = mlm.fill_mask(masked_sentence, topk=top)

        max_len = 0
        word_batch = []
        ids_batch = []
        attention_mask_batch = []
        verb_posi_batch = []

        for each in mlm_result:
            words[verb_posi] = each[2].strip()
            if words[verb_posi] in punctuation:
                continue

            pos = nltk.pos_tag(words)
            if pos[verb_posi][1] not in ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]:
                continue

            tokens = []
            for i, word in enumerate(words):
                token_tmp = tokenizer.tokenize(word)
                if i == verb_posi:
                    new_verb_posi = len(tokens)
                tokens += token_tmp
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            new_verb_posi += 1

            ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(ids)
            max_len = max(max_len, len(ids))

            word_batch.append(words[verb_posi])
            ids_batch.append(ids)
            attention_mask_batch.append(attention_mask)
            verb_posi_batch.append(new_verb_posi)

        ids_batch = np.array([each + [0] * (max_len-len(each)) for each in ids_batch], dtype=int)
        attention_mask_batch = np.array([each + [0] * (max_len-len(each)) for each in attention_mask_batch], dtype=int)
        verb_posi_batch = np.array(verb_posi_batch, dtype=int)

        ids_batch = torch.tensor(ids_batch, dtype=torch.int64).cuda()
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.int64).cuda()
        verb_posi_batch = torch.tensor(verb_posi_batch, dtype=torch.int64).cuda()

        outputs = model(ids_batch, attention_mask=attention_mask_batch, word_posi=verb_posi_batch)
        ml_score = outputs[0].detach().sigmoid().view(-1)
        ml_pred = torch.argsort(ml_score, descending=True).detach().cpu().numpy()
        ml_score = ml_score.cpu().numpy()
        f_ans.write("sentence: %s\nverb: %s\nprediction: " % (sentence, verb))
        result = [sentence, verb, verb_posi, []]

        for i in range(len(ids_batch)):
            f_ans.write("%s(%.6f, %d), " % (word_batch[ml_pred[i]], ml_score[ml_pred[i]], ml_pred[i]))
            result[3].append((word_batch[ml_pred[i]], float(ml_score[ml_pred[i]])))
        f_ans.write("\n\n")
        results.append(result)
    return results


def main(input_file=None, output_dir=None, model_file=None, gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu if gpu else args.gpu

    if output_dir:
        dir_path = output_dir
    else:
        dir_name = time.strftime(r"%Y%m%d-%H%M")
        dir_path = os.path.join(os.getcwd(), "result", dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    f_mlm_ans = open(os.path.join(dir_path, "mlm_ans.txt"), 'w')
    f_mlm_json = open(os.path.join(dir_path, "mlm_ans.json"), 'w')

    with open(input_file if input_file else args.input_file, "r", encoding="utf-8") as f:
        data = loadData(f)

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    # roberta = RobertaModel.from_pretrained('/home/zhangxurui/roberta.large', checkpoint_file='model.pt')

    model = BertForMD.from_pretrained(model_name, num_labels=1)
    model.load_state_dict(torch.load(model_file if model_file else args.model_file))

    model.cuda()

    roberta.eval()
    model.eval()
    with torch.no_grad():
        results = mlm_fn(model, roberta, data, f_mlm_ans)
    json.dump(results, f_mlm_json, indent=4)


if __name__ == "__main__":
    main()
