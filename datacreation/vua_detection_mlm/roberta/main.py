import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from model import RobertaForMD

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epoch_num", type=int)

parser.add_argument("--message", "-m", type=str, default="")
parser.add_argument("--warmup_ratio", type=float)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--gpu", type=str)

args = parser.parse_args()

if args.seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model_name = "roberta-large"

tokenizer = RobertaTokenizer.from_pretrained(model_name)

def loadMOH(f):
    instances = []
    verb = ""
    f.readline()
    f_csv = csv.reader(f)
    for line in f_csv:
        verb = line[2]
        if verb not in tokenizer.vocab:
            continue
        sentence = line[3]
        words = sentence.strip().split(" ")
        verb_posi = int(line[4])
        label = int(line[5])

        new_verb_posi = 0
        tokens = []
        for i, word in enumerate(words):
            word = " " + word if i else word
            tmp_tokens = tokenizer.tokenize(word)
            if i == verb_posi:
                new_verb_posi = len(tokens)
            tokens += tmp_tokens

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        new_verb_posi += 1

        ids = tokenizer.convert_tokens_to_ids(tokens)

        instances.append((ids, label, verb, new_verb_posi, sentence))
    return instances


def loadVUA(f):
    instances = []
    f.readline()
    f_csv = csv.reader(f)
    for line in tqdm(f_csv):
        verb = line[2]
        sentence = line[3].split(" ")
        verb_posi = int(line[4])
        label = int(line[5])

        tokens = []
        new_verb_posi = 0
        for i in range(len(sentence)):
            word = " " + sentence[i] if i else sentence[i]
            token_tmp = tokenizer.tokenize(sentence[i])
            if i == verb_posi:
                new_verb_posi = len(tokens)
            tokens += token_tmp

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        new_verb_posi += 1

        ids = tokenizer.convert_tokens_to_ids(tokens)

        instances.append((ids, label, verb, new_verb_posi, line[3]))

    return instances


def collate_fn(batch):
    max_len = max([len(each[0]) for each in batch])
    ids = np.array([each[0] + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    attention_mask = np.array([[1] * len(each[0]) + [0] * (max_len-len(each[0])) for each in batch], dtype=int)
    label = np.array([each[1] for each in batch], dtype=int)
    verb = [each[2] for each in batch]
    verb_posi = np.array([each[3] for each in batch], dtype=int)
    sentence = [each[4] for each in batch]

    ids = torch.tensor(ids, dtype=torch.int64).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).cuda()
    label = torch.tensor(label, dtype=torch.int64).cuda()
    verb_posi = torch.tensor(verb_posi, dtype=torch.int64).cuda()

    return ids, attention_mask, label, verb, verb_posi, sentence


def metaphor_ans(pred, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        elif labels[i] == 0:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def prf(tp, tn, fp, fn):
    try:
        accu = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("Division by zero.\n")
        return (0, 0, 0, 0)

    return (accu, precision, recall, f_score)


def train_fn(model, dataloader, optimizer, scheduler, f_log, f_ans=None):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tot = 0
    loss_sum = 0
    for ids, attention_mask, label, verb, verb_posi, sentence in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(ids, attention_mask=attention_mask, labels=label.to(torch.float32), word_posi=verb_posi)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
        print(loss.item())

        score = outputs[1].detach().sigmoid().cpu()
        pred = outputs[1].detach().sigmoid().gt(0.5).cpu().to(torch.int64).numpy()
        label = label.detach().cpu().numpy()

        _tp, _tn, _fp, _fn = metaphor_ans(pred, label)
        tp += _tp
        tn += _tn
        fp += _fp
        fn += _fn
        tot += pred.shape[0]

        if f_ans:
            for i in range(ids.shape[0]):
                f_ans.write("sentence: %s\nverb: %s\nlabel: %s, prediction: %s(scores: %.6f)\n\n" % (sentence[i], verb[i], "metaphorical" if label[i] else "literal", "metaphorical" if pred[i] else "literal", score[i]))

    print(tp, tn, fp, fn)
    accu, precision, recall, f_score = prf(tp, tn, fp, fn)
    loss_sum /= tot
    print("loss: %6f\naccu: %6f\nprecision: %6f\nrecall: %6f\nf1: %6f\n" % (loss_sum, accu, precision, recall, f_score))
    f_log.write("loss=%6f, accu=%6f, precision=%6f, recall=%6f, f1=%6f\n" % (loss_sum, accu, precision, recall, f_score))


def evaluate_fn(model, dataloader, f_log, f_ans=None):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tot = 0
    loss_sum = 0
    for ids, attention_mask, label, verb, verb_posi, sentence in tqdm(dataloader):
        outputs = model(ids, attention_mask=attention_mask, labels=label.to(torch.float32), word_posi=verb_posi)
        loss = outputs[0]
        loss_sum += loss.item()
        print(loss.item())

        score = outputs[1].detach().sigmoid().cpu()
        pred = outputs[1].detach().sigmoid().gt(0.5).cpu().to(torch.int64).numpy()
        label = label.detach().cpu().numpy()

        _tp, _tn, _fp, _fn = metaphor_ans(pred, label)
        tp += _tp
        tn += _tn
        fp += _fp
        fn += _fn
        tot += pred.shape[0]

        if f_ans:
            for i in range(ids.shape[0]):
                f_ans.write("sentence: %s\nverb: %s\nlabel: %s, prediction: %s(scores: %.6f)\n\n" % (sentence[i], verb[i], "metaphorical" if label[i] else "literal", "metaphorical" if pred[i] else "literal", score[i]))

    print(tp, tn, fp, fn)
    accu, precision, recall, f_score = prf(tp, tn, fp, fn)
    loss_sum /= tot
    print("loss: %6f\naccu: %6f\nprecision: %6f\nrecall: %6f\nf1: %6f\n" % (loss_sum, accu, precision, recall, f_score))
    f_log.write("loss=%6f, accu=%6f, precision=%6f, recall=%6f, f1=%6f\n" % (loss_sum, accu, precision, recall, f_score))


def mlm_fn(model, mlm, dataloader, f_ans, top=200):
    for ids, attention_mask, label, verb, verb_posi, sentence in tqdm(dataloader):
        for i in range(ids.shape[0]):
            ids[i][verb_posi[i]] = tokenizer.mask_token_id
        pred_score = mlm(ids, attention_mask=attention_mask)[0]
        for i in range(pred_score.shape[0]):
            pred = torch.argsort(pred_score[i][verb_posi[i]], descending=True).detach().cpu().numpy().tolist()[:top]
            new_ids = ids[i].repeat(top, 1)
            for j in range(top):
                new_ids[j][verb_posi[i]] = pred[j]
            outputs = model(new_ids, attention_mask=attention_mask[i].repeat(top, 1), word_posi=verb_posi[i].repeat(top))
            ml_score = outputs[0].detach().sigmoid().view(-1)
            ml_pred = torch.argsort(ml_score, descending=True).detach().cpu().numpy()
            ml_score = ml_score.cpu().numpy()
            f_ans.write("sentence: %s\nverb: %s\nlabel: %s\nprediction: " % (sentence[i], verb[i], "metaphorical" if label[i] else "literal"))

            for j in range(top):
                token = tokenizer.convert_ids_to_tokens(pred[ml_pred[j]])
                if token.startswith("Ġ"):
                    token = token[1:]
                f_ans.write("%s(%.6f, %d), " % (token, ml_score[ml_pred[j]], ml_pred[j]))
            f_ans.write("\n\n")


def main():
    dir_name = time.strftime(r"%Y%m%d-%H%M")
    dir_path = os.path.join(os.getcwd(), "result", dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    log = open(os.path.join(dir_path, "log.txt"), 'w')
    f_train_ans = open(os.path.join(dir_path, "train_ans.txt"), 'w')
    f_valid_ans = open(os.path.join(dir_path, "valid_ans.txt"), 'w')
    f_mlm_ans = open(os.path.join(dir_path, "mlm_ans.txt"), 'w')
    log.write(args.message + "\n\n")

    log.write("learning rate: %f\nbatch size: %d\nepoch num: %d\nmodel name: %s\n" % (args.learning_rate, args.batch_size, args.epoch_num, model_name))


    with open("../../data/MOH/MOH_for_mlm.csv", "r", encoding="utf-8") as f:
        MOH_for_mlm = loadMOH(f)
    mlm_dataloader = torch.utils.data.DataLoader(MOH_for_mlm, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    with open("../../data/VUA_verb/data_augmentation_train.csv", "r", encoding="utf-8") as f:
        train_aug = loadVUA(f)
    with open("../../data/VUA_verb/VUA_formatted_train_noVAL.csv", "r", encoding="utf-8") as f:
        train_ori = loadVUA(f)
    train = train_aug + train_ori
    with open("../../data/VUA_verb/VUA_formatted_test.csv", "r", encoding="utf-8") as f:
        test = loadVUA(f)
    train_vua = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    test_vua = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    mlm = RobertaForMaskedLM.from_pretrained(model_name)
    model = RobertaForMD.from_pretrained(model_name, num_labels=1)

    mlm.cuda()
    model.cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    t_total = len(train)//args.batch_size*args.epoch_num
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio*t_total, num_training_steps=t_total)

    for epoch in range(args.epoch_num):
        log.write("epoch: %d\n" % epoch)
        log.write("train: ")
        model.train()
        if epoch == args.epoch_num - 1:
            train_fn(model, train_vua, optimizer, scheduler, log, f_train_ans)
        else:
            train_fn(model, train_vua, optimizer, scheduler, log)

        log.write("valid: ")
        model.eval()
        with torch.no_grad():
            if epoch == args.epoch_num - 1:
                evaluate_fn(model, test_vua, log, f_valid_ans)
            else:
                evaluate_fn(model, test_vua, log)

    mlm.eval()
    with torch.no_grad():
        mlm_fn(model, mlm, mlm_dataloader, f_mlm_ans)


if __name__ == "__main__":
    main()
