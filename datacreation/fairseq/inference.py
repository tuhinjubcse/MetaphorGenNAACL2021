import argparse

import sys
import os
import copy
import time
import torch
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="0"


parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='./metaphor.source', help='input file')
parser.add_argument('--outfile', type=str, default='./metaphor.hypo', help='output file')
parser.add_argument('--apply_disc', action='store_true', help='whether to use discriminators to rescore')
parser.add_argument('--scorers', type=str, default='./WP_scorers.tsv', help='tsv with discriminator info')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dedup', action='store_false')
parser.add_argument('--banned_tok', nargs='+', default=["[", " [", "UN", " UN"], help="tokens to prevent generating")
parser.add_argument('--max_len', type=int, default=250, help="max length of generation in BPE tok") 


def load_scorers(filepath):
    scorer_config, coefs, model_info = [], [], []
    with open(filepath, "r") as scorer_file:
        for line in scorer_file:
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t') # expect coef and then name
            coefs.append(float(fields[0]))
            model_info.append(fields[1:])
            scorer_config.append(fields)
    return coefs, model_info, scorer_config

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

bart = BARTModel.from_pretrained(
    'checkpoint-metaphor/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='metaphor'
)


bart.eval()

if use_cuda:
    bart.cuda() # remove this line if not running with cuda
    bart.half() # doesn't work with CPU

np.random.seed(4)
torch.manual_seed(4)



### load discriminators
scorers, coefs = [], []
apply_disc = True
if apply_disc:
    coefs, scorer_info, scorer_config = load_scorers(args.scorers)
    for info in scorer_info:
        if len(info) > 3:
            print("too many fields (3 req): {}".format(info))
        model_dir, checkpoint_name, data_path = info

        roberta = RobertaModel.from_pretrained(
            model_dir,
            checkpoint_file=checkpoint_name,data_name_or_path=data_path)

        roberta.eval()
        if use_cuda:
            roberta.cuda()
            roberta.half()

            scorers.append(roberta)


count = 1
bsz = args.batch_size
pad_toks = {0} #, 2}
banned_verbs, banned_ids = [], []

# if args.dedup:
#     verb_strings = ["<V>", " <V>"]
#     verb_tensors = [bart.encode(v) for v in verb_strings]
#     banned_verbs = list(set([i.data.item() for t in verb_tensors for i in t]) - pad_toks)

# if args.banned_tok:
#     banned_tok_tensors = [bart.encode(t) for t in args.banned_tok]
#     banned_ids = list(set([i.data.item() for t in banned_tok_tensors for i in t]) - pad_toks)


with open(args.infile, 'r') as fin, open(args.outfile, 'w') as fout:
    sline = fin.readline().strip()
    slines = [sline]
    print("Example Data: {}".format(sline.strip()))
    for sline in fin:
        if count % bsz == 0 and count:
            start_time = time.time()
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, lenpen=2.0,
                                               max_len_b=30, min_len=7, no_repeat_ngram_size=3,
                                               rescore=True,
                                               coefs=coefs, scorers=scorers, dedup=args.dedup, 
                                               banned_toks=banned_ids, verb_idxs=banned_verbs)
            elapsed = time.time() - start_time
            print("Seconds per batch: {}".format(elapsed))
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', '') + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        with torch.no_grad():
            hypotheses_batch = bart.sample(slines, lenpen=2.0,
                                               max_len_b=30, min_len=7, no_repeat_ngram_size=3,
                                               rescore=True,
                                               coefs=coefs, scorers=scorers, dedup=args.dedup,
                                               banned_toks=banned_ids, verb_idxs=banned_verbs)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()
