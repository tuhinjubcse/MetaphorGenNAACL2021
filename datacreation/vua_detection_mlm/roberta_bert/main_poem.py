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

from .model import BertForMD

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str)
parser.add_argument("--model_file", type=str)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--batch_size", type=str, default=32)
parser.add_argument("--threshold", type=float, default=0.95)

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
    instances = []
    f.readline()
    f_csv = csv.reader(f)
    for line in tqdm(f_csv):
        sentence = line[0]
        verb = line[1]
        verb_posi = int(line[2])
        words = sentence.split(" ")
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

        instances.append((ids, new_verb_posi, sentence, verb, verb_posi))
        # if len(instances) > 5:
        #     break

    return instances


def loadMetaphorData(f, threshold):
    instances = []
    f_csv = csv.reader(f)
    for line in tqdm(f_csv):
        sentence = line[0].split(" ")
        verb = line[1]
        verb_posi = int(line[2])
        score = float(line[3])
        if score < threshold:
            continue

        tokens = []
        new_verb_posi = 0
        for i in range(len(sentence)):
            token_tmp = tokenizer.tokenize(sentence[i])
            if i == verb_posi:
                new_verb_posi = len(tokens)
            tokens += token_tmp

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        new_verb_posi += 1

        ids = tokenizer.convert_tokens_to_ids(tokens)

        instances.append((ids, 1, verb, new_verb_posi, line[0], verb_posi))

    return instances


def collate_fn(batch):
    max_len = max(len(each[0]) for each in batch)
    ids = np.array([each[0] + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    attention_mask = np.array([[1] * len(each[0]) + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    new_verb_posi = np.array([each[1] for each in batch], dtype=int)
    sentence = [each[2] for each in batch]
    verb = [each[3] for each in batch]
    verb_posi = [each[4] for each in batch]

    ids = torch.tensor(ids, dtype=torch.int64).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).cuda()
    new_verb_posi = torch.tensor(new_verb_posi, dtype=torch.int64).cuda()

    return ids, attention_mask, new_verb_posi, sentence, verb, verb_posi


def collate_mlm_fn(batch):
    max_len = max([len(each[0]) for each in batch])
    ids = np.array([each[0] + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    attention_mask = np.array([[1] * len(each[0]) + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    label = np.array([each[1] for each in batch], dtype=int)
    verb = [each[2] for each in batch]
    new_verb_posi = np.array([each[3] for each in batch], dtype=int)
    sentence = [each[4] for each in batch]
    verb_posi = [each[5] for each in batch]

    ids = torch.tensor(ids, dtype=torch.int64).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).cuda()
    label = torch.tensor(label, dtype=torch.int64).cuda()
    new_verb_posi = torch.tensor(new_verb_posi, dtype=torch.int64).cuda()

    return ids, attention_mask, label, verb, new_verb_posi, sentence, verb_posi


def test_fn(model, dataloader, f_csv):
    for ids, attention_mask, new_verb_posi, sentence, verb, verb_posi in tqdm(dataloader):
        outputs = model(ids, attention_mask=attention_mask, word_posi=new_verb_posi)
        # loss = outputs[0]
        # loss_sum += loss.item()
        # print(loss.item())

        score = outputs[0].detach().sigmoid().cpu().tolist()

        data = [[sentence[i], verb[i], verb_posi[i], score[i][0]] for i in range(ids.shape[0])]
        f_csv.writerows(data)


def mlm_fn(model, mlm, dataloader, f_ans, top=200):
    results = []
    for ids, attention_mask, label, verb, token_verb_posi, sentence, verb_posi in tqdm(dataloader):
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
            ml_pred = torch.argsort(ml_score).detach().cpu().numpy()
            ml_score = ml_score.cpu().numpy()
            f_ans.write("sentence: %s\nverb: %s\nlabel: %s\nprediction: " % (sentence[i], verb[i], "metaphorical" if label[i] else "literal"))
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


def main(threshold=None, input_file=None, output_dir=None, model_file=None, gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu if gpu else args.gpu

    if output_dir:
        dir_path = output_dir
    else:
        dir_name = time.strftime(r"%Y%m%d-%H%M")
        dir_path = os.path.join(os.getcwd(), "result", dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # metaphor detection
    f = open(os.path.join(dir_path, "poem_metaphor_score.csv"), "w")
    f_csv = csv.writer(f)

    model = BertForMD.from_pretrained(model_name, num_labels=1)
    model.load_state_dict(torch.load(model_file))
    model.cuda()

    with open(input_file if input_file else args.input_file, "r", encoding="utf-8") as f:
        dataset = loadData(f)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        test_fn(model, dataloader, f_csv)

    f.close()

    # generate literal counterpart
    f_mlm_ans = open(os.path.join(dir_path, "mlm_ans.txt"), 'w')
    f_mlm_json = open(os.path.join(dir_path, "mlm_ans.json"), 'w')

    with open(os.path.join(dir_path, "poem_metaphor_score.csv"), "r") as f:
        poem_for_mlm = loadMetaphorData(f, threshold if threshold else args.threshold)
    mlm_dataloader = torch.utils.data.DataLoader(poem_for_mlm, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_mlm_fn)

    mlm = BertForMaskedLM.from_pretrained(model_name)
    mlm.cuda()
    mlm.eval()

    with torch.no_grad():
        results = mlm_fn(model, mlm, mlm_dataloader, f_mlm_ans)
    json.dump(results, f_mlm_json, indent=4)


if __name__ == "__main__":
    main()
