""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for a , b in zip(dataset, predictions):
        total = total+1
        ground_truths = [a]
        prediction = b
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)


    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


gold = []
pred = []
m1={}
m2 = {}

# for line in open('generated.txt'):
#     line = line.split(' [SEP] ')
#     m1[line[0].rstrip()]=line[1].strip()

# for line in open('val.target'):
#     line = line.split(' [SEP] ')
#     m2[line[0].rstrip()]=line[1].strip()

# questions = {}
# for line in open('questions.txt'):
#     questions[line.strip()]=True

# f = open('generated_answers.txt','w')
# for line1,line2 in zip(open('gold_ans.txt'),open('pred_rouge_ans.txt')):
#     line = line1.split(' [SEP] ')
#     q = line[0].rstrip()
#     if q in questions:
#         line1 = line1.strip().split(' [SEP] ')[1]
#         line2 = line2.strip().split(' [SEP] ')[1]
#         gold.append(line1)
#         pred.append(line2)
    # q = line
    # f.write(q+' [SEP] '+m[q]+'\n')

# for line in open('questions.txt'):
#     if line.strip() in m1 and line.strip() in m2:
#         gold.append(m2[line.strip()])
#         pred.append(m1[line.strip()])

m = {}
f = open('supportdoc_recall.txt','w')
for line1,line2 in zip(open('./ELI5/val.source'),open('./ELI5/val.target')):
    gold,pred = [],[]
    if line1!='':
        line = line1.strip().split(' <EOT> ')[1]
        line2 = line2.strip()
        gold.append(line2)
        pred.append(line)
        f.write(line1.strip().split(' <EOT> ')[0]+'\t'+str(evaluate(gold,pred)['f1'])+'\n')

#print(evaluate(gold,pred))



