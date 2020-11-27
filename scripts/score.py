#!/usr/bin/env python3

'''
@Time   : 2019-04-04 14:38:55
@Author : su.zhu
@Desc   : 
'''

import argparse
import prettytable
import json

parser = argparse.ArgumentParser()

parser.add_argument('--annotation', required=True, help='dataset file with semantic annotation')
parser.add_argument('--prediction', required=True, help='dataset file with predicted semantics')

opt = parser.parse_args()


## read input dataset
with open(opt.annotation) as in_file:
    slu_anno_data = json.load(in_file)
with open(opt.prediction) as in_file:
    slu_pred_data = json.load(in_file)

assert len(slu_anno_data) == len(slu_pred_data)

total_utter_number = 0
correct_utter_number = 0
TP, FP, FN = 0, 0, 0
for anno_dialogue, pred_dialogue in zip(slu_anno_data, slu_pred_data):
    assert len(anno_dialogue['utterances']) == len(pred_dialogue['utterances'])
    for anno_utterance, pred_utterance in zip(anno_dialogue['utterances'], pred_dialogue['utterances']):
        anno_semantics = anno_utterance["semantic"]
        pred_semantics = pred_utterance["semantic"]
        anno_semantics = set([tuple(item) for item in anno_semantics])
        pred_semantics = set([tuple(item) for item in pred_semantics])

        total_utter_number += 1
        if anno_semantics == pred_semantics:
            correct_utter_number += 1

        TP += len(anno_semantics & pred_semantics)
        FN += len(anno_semantics - pred_semantics)
        FP += len(pred_semantics - anno_semantics)

table = prettytable.PrettyTable(["metrics", "scores (%)"])
table.add_row(["Precision", "%.2f" % (100 * TP / (TP + FP))])
table.add_row(["Recall", "%.2f" % (100 * TP / (TP + FN))])
table.add_row(["F1-score", "%.2f" % (100 * 2 * TP / (2 * TP + FN +FP))])
table.add_row(["Joint accuracy", "%.2f" % (100 * correct_utter_number / total_utter_number)])

print(table)
