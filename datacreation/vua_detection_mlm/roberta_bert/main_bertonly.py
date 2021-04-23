import argparse
import csv
import json
import os
import random
import time
from string import punctuation

import nltk
import numpy as np
import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup

from model import BertForMD

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str)
parser.add_argument("--model_file", type=str)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--batch_size", type=str, default=32)

args = parser.parse_args()

if args.seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
                data.append([line, words, verb, i])

    instances = []
    for sentence, words, verb, verb_posi in data:
        tokens = []
        new_verb_posi = 0
        for i in range(len(words)):
            token_tmp = tokenizer.tokenize(words[i])
            if i == verb_posi:
                new_verb_posi = len(tokens)
            tokens += token_tmp

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        new_verb_posi += 1

        ids = tokenizer.convert_tokens_to_ids(tokens)

        instances.append((ids, verb, new_verb_posi, sentence, verb_posi))
    return instances


def collate_fn(batch):
    max_len = max([len(each[0]) for each in batch])
    ids = np.array([each[0] + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    attention_mask = np.array([[1] * len(each[0]) + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    verb = [each[1] for each in batch]
    new_verb_posi = np.array([each[2] for each in batch], dtype=int)
    sentence = [each[3] for each in batch]
    verb_posi = [each[4] for each in batch]

    ids = torch.tensor(ids, dtype=torch.int64).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).cuda()
    new_verb_posi = torch.tensor(new_verb_posi, dtype=torch.int64).cuda()

    return ids, attention_mask, verb, new_verb_posi, sentence, verb_posi


def mlm_fn(model, mlm, dataloader, f_ans, top=200):
    results = []
    for ids, attention_mask, verb, token_verb_posi, sentence, verb_posi in tqdm(dataloader):
        for i in range(ids.shape[0]):
            ids[i][token_verb_posi[i]] = tokenizer.mask_token_id
        pred_score = mlm(ids, attention_mask=attention_mask)[0]
        for i in range(pred_score.shape[0]):
            pred = torch.argsort(pred_score[i][token_verb_posi[i]], descending=True).detach().cpu().numpy().tolist()[:top]
            new_ids = ids[i].repeat(top, 1)
            for j in range(top):
                new_ids[j][token_verb_posi[i]] = pred[j]
            outputs = model(new_ids, attention_mask=attention_mask[i].repeat(top, 1), word_posi=token_verb_posi[i].repeat(top))
            ml_score = outputs[0].detach().sigmoid().view(-1)
            ml_pred = torch.argsort(ml_score, descending=True).detach().cpu().numpy()
            ml_score = ml_score.cpu().numpy()
            f_ans.write("sentence: %s\nverb: %s\nprediction: " % (sentence[i], verb[i]))
            result = [sentence[i], verb[i], verb_posi[i], []]

            words = sentence[i].split(" ")
            for j in range(top):
                token = tokenizer.convert_ids_to_tokens(pred[ml_pred[j]])
                # remove non-verb
                if token.startswith("#"):
                    continue
                if token in punctuation:
                    continue
                words[verb_posi[i]] = token
                pos = nltk.pos_tag(words)
                if pos[verb_posi[i]][1] not in ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]:
                    continue

                f_ans.write("%s(%.6f, %d), " % (token, ml_score[ml_pred[j]], ml_pred[j]))
                result[3].append((token, float(ml_score[ml_pred[j]])))
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

    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    mlm = BertForMaskedLM.from_pretrained(model_name)
    mlm.cuda()

    model = BertForMD.from_pretrained(model_name, num_labels=1)
    model.load_state_dict(torch.load(model_file if model_file else args.model_file))

    model.cuda()

    mlm.eval()
    model.eval()
    with torch.no_grad():
        results = mlm_fn(model, mlm, dataloader, f_mlm_ans)
    json.dump(results, f_mlm_json, indent=4)


if __name__ == "__main__":
    main()
