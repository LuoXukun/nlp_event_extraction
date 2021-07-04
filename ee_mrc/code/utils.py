"""
Author: Xinrui Ma
Date: 2021-06-26 16:00:50
LastEditTime: 2021-06-29 19:31:17
Description: Some utils functions used in EE work
"""

import config
import json
import joblib
import random
import numpy as np
from collections import defaultdict
import datetime
import torch
import pandas as pd
import csv
from transformers import BertTokenizer




def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def search(pattern, sequence):
    """ 从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding=encoding) as infile:
        for line in infile:
            result.append(line.strip())
    return result


def calculate_token_cpg(p, g):
    """ p: predict position (start, end) [(26, 30)]
        g: gold position (start, end) [(32, 33)]
        return number of correct, predict, golden [3, 5, 2]
    """
    lenc, lenp, leng = 0, 0, 0
    if p:
        for (a, b) in p:
            lenp += b - a + 1
    if g:
        for (a, b) in g:
            leng += b - a + 1
    if p and g:
        for (a, b) in p:
            for (c, d) in g:
                lenc += max(min(b, d) - max(a, c) + 1, 0)
    return lenc, lenp, leng


def make_text_map(data_path):
    """ map text_id to text
    """
    text_map = {}
    lines = read_by_lines(data_path)
    for line in lines:
        line = json.loads(line.strip())
        text_map[line["id"]] = line["text"]
    return text_map


def calculte_ner(_tok_text_ids, _mask, _pred, _orig_argum, id2role):
    from seqlabel import TOKENIZER
    tokenizer = TOKENIZER
    pair = []; idx = 0
    while idx < _mask:
        if _pred[idx] % 2 == 1:
            start = idx; end = idx + 1
            while end < _mask:
                if _pred[end] == _pred[start] + 1:
                    end += 1
                else:
                    break
            idx = end; pair.append((start, end, _pred[start]))
        else:
            idx += 1

    _pred_argum = {}
    for x, y, z in pair:
        output = tokenizer.decode(_tok_text_ids[x: y])
        output = "".join(str(output).split())
        argu_role = id2role[z // 2 + 1]
        _pred_argum[output] = argu_role
    b = len(_pred_argum); c = len(_orig_argum)
    correct_argum = {}
    for kk in _pred_argum:
        if kk in _orig_argum and _pred_argum[kk] == _orig_argum[kk]:
            correct_argum[kk] = _pred_argum[kk]
    a = len(correct_argum)
    return a, b, c, _pred_argum


def load_schema():
    event_type_to_role = defaultdict(set)
    lines = read_by_lines(config.EVENT_SCHEMA_FILE)
    for line in lines:
        line = json.loads(line.strip())
        for role in line["role_list"]:
            event_type_to_role[line["event_type"]].add(role["role"])

    print("Event Schema has been loaded.\n")
    return event_type_to_role


def make_evt_map():
    event_type_to_role = load_schema()
    evt = event_type_to_role.keys()
    evt2id, id2evt = {"O": 0}, {0: "O"}
    for i, e in enumerate(evt, 1):
        evt2id[e] = i
        id2evt[i] = e
    meta_data = {
        "evt2id": evt2id,
        "id2evt": id2evt
    }
    joblib.dump(meta_data, config.EVENT_TYPE_FILE)
    print("Use Event Type Map.")
    
    
def make_role_map():
    """ 论元角色的组合标签
    """
    event_type_to_role = load_schema()
    role2id, id2role = {"O": 0}, {0: "O"}
    idx = 1
    for k in event_type_to_role:
        for r in event_type_to_role[k]:
            role = k + "-" + r
            role2id[role] = idx; id2role[idx] = role; idx += 1
    meta_data = {
        "role2id": role2id,
        "id2role": id2role
    }
    joblib.dump(meta_data, config.ROLE_MAP_FILE)
    print("Use Role Map, total role tag {}.\n".format(len(role2id)))
            
        
def make_single_role_map():
    """ 论元角色的单独标签
    """
    role_set = set()
    lines = read_by_lines(config.EVENT_SCHEMA_FILE)
    for line in lines:
        line = json.loads(line.strip())
        for role in line["role_list"]:
            role_set.add(role["role"])
    print(sorted(role_set))
    role2id, id2role = {"O": 0}, {0: "O"}
    for idx, role in enumerate(sorted(role_set), 1):
        role2id[role] = idx; id2role[idx] = role
    print(role2id)
    print(id2role)
    meta_data = {
        "role2id": role2id,
        "id2role": id2role
    }
    joblib.dump(meta_data, config.SINGLE_ROLE_MAP_FILE)
    print("Use Single Role Map, total role tag {}.\n".format(len(role2id))) 


def load_data_event_type(in_path):
    lines = read_by_lines(in_path)
    random.seed(2021)
    random.shuffle(lines)
    sentences, tag, text_id = [], [], []
    for line in lines:
        line = json.loads(line.strip())
        if not line: continue
        sentences.append(line["text"])
        tag.append(line["tag"])
        text_id.append(line["text_id"])
    print("There are {} sentences in {}.\n".format(len(tag), in_path))
    return sentences, tag, text_id


def to_list_one(tensor):
    """ return index list where tensor element is 1, discard 0 values,
    """
    # return (tensor).nonzero().flatten().detach().tolist()
    return torch.nonzero(tensor, as_tuple =False).flatten().detach().tolist()


def calculate_cpg(input):
    """ return number of correct, predict, gold and positions
        used for start/end 
    """
    p1, p2, g1, g2 = [to_list_one(e) for e in input]

    correct = 0
    gold_zip = list(zip(g1, g2))  # [(12, 15)]

    # pred has no start/end
    if len(p1) * len(p2) == 0:
        return 0, 0, len(gold_zip), [], gold_zip

    else:
        pred_zip = list(zip(p1, p2))
        for pair in pred_zip:
            if pair in gold_zip:
                correct += 1
        return correct, len(pred_zip), len(gold_zip), pred_zip, gold_zip



def calculate_cpg_io(input):
    """ return number of correct, predict, gold and positions
        used for in/out
    """
    def return_zip_start_end(labels):
        pair = []
        ix = 0
        while ix < len(labels):
            if labels[ix]:
                s, e = ix, ix + 1
                while e < len(labels) and labels[e]:
                    e += 1
                ix = e
                pair.append((s, e - 1))
            else:
                ix += 1
        return pair
    
    pred_zip = return_zip_start_end(input[0])
    gold_zip = return_zip_start_end(input[1])
    correct = 0
    # pred has no start/end
    if not pred_zip:
        return 0, 0, len(gold_zip), [], gold_zip

    else:
        for pair in pred_zip:
            if pair in gold_zip:
                correct += 1
        return correct, len(pred_zip), len(gold_zip), pred_zip, gold_zip


def print_final_output(in_path, out_path):
    """ print tokens, using start and end positions
    """
    tokenizer = BertTokenizer.from_pretrained("../input/chinese-bert-wwm-ext/")
    df_ensembel = pd.read_csv(in_path).reset_index(drop=True)
    final_print = []
    text_map = make_text_map(config.TESTING_FILE)
    for row in df_ensembel.itertuples():
        context_id = getattr(row, 'context_id')
        context = text_map[context_id]
        role = getattr(row, 'role')
        pred = getattr(row, 'pred_position')
        gold = getattr(row, 'gold_position')

        # question
        q_tokens = tokenizer.tokenize(role)
        q_token_ids = tokenizer.convert_tokens_to_ids(q_tokens)
        # context
        c_tokens = tokenizer.tokenize(context)
        c_token_ids = tokenizer.convert_tokens_to_ids(c_tokens)

        # [CLS] question [SEP] context [SEP]
        input_ids = [101] + q_token_ids + [102] + c_token_ids + [102]

        predict_list, golden_list = [], []
        if pred:
            for (s, e) in eval(pred):
                ans_ids = input_ids[s: e + 1]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                predict_list.append("".join(ans))
        if gold:
            for (s, e) in eval(gold):
                ans_ids = input_ids[s: e + 1]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                golden_list.append("".join(ans))

        output = {"context_id": context_id, "role": role, "pred": predict_list, "gold": golden_list}
        final_print.append(output)

    csv_columns = ["context_id", "role", "pred", "gold"]
    with open(out_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for d in final_print:
            writer.writerow(d)
    print("Final print saved in {}".format(out_path))


def print_final_output_io(in_path, out_path):
    """ print tokens, using start and end positions
    """
    tokenizer = BertTokenizer.from_pretrained("../input/chinese-bert-wwm-ext/")
    df_ensembel = pd.read_csv(in_path).reset_index(drop=True)
    final_print = []
    text_map = make_text_map(config.TESTING_FILE)
    for row in df_ensembel.itertuples():
        context_id = getattr(row, 'context_id')
        context = text_map[context_id]
        role = getattr(row, 'role')
        pred = getattr(row, 'pred_position')
        gold = getattr(row, 'gold_position')

        # question
        q_tokens = tokenizer.tokenize(role)
        q_token_ids = tokenizer.convert_tokens_to_ids(q_tokens)
        # context
        c_tokens = tokenizer.tokenize(context)
        c_token_ids = tokenizer.convert_tokens_to_ids(c_tokens)

        # [CLS] question [SEP] context [SEP]
        input_ids = [101] + q_token_ids + [102] + c_token_ids + [102]

        lenq = len(q_token_ids) + 2

        predict_list, golden_list = [], []
        if pred:
            for (s, e) in eval(pred):
                ans_ids = input_ids[s + lenq: e + 1 + lenq]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                predict_list.append("".join(ans))
        if gold:
            for (s, e) in eval(gold):
                ans_ids = input_ids[s + lenq: e + 1 + lenq]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                golden_list.append("".join(ans))

        output = {"context_id": context_id, "role": role, "pred": predict_list, "gold": golden_list}
        final_print.append(output)

    csv_columns = ["context_id", "role", "pred", "gold"]
    with open(out_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for d in final_print:
            writer.writerow(d)
    print("Final print saved in {}".format(out_path))


if __name__ == "__main__":
    # print_final_output(in_path="../output/mrc_se.csv", out_path="../output/mrc_se_print.csv")
    print_final_output_io(in_path="../output/mrc_io.csv", out_path="../output/mrc_io_print.csv")

