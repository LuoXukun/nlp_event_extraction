#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.25

import json
import argparse

def get_event_role_list(result, type="string"):
    events, roles = set(), []
    for event in result["event_list"]:
        event_type = event["event_type"]
        events.add(event_type)
        for argument in event["arguments"]:
            if type == "string":
                roles.append(str((event_type, argument["role"], argument["argument"])))
            elif type == "tuple":
                roles.append((event_type, argument["role"], argument["argument"]))
    return list(events), roles

def evaluate(test_path, gold_path, parse_path):
    ft = open(test_path, "r", encoding="utf-8")
    fg = open(gold_path, "r", encoding="utf-8")

    correct, gold_number, pred_number = 0, 0, 0
    event_correct, event_gold, event_pred = 0, 0, 0

    if parse_path is not None:
        fw = open(parse_path, "w", encoding="utf-8")

    for test_line, gold_line in zip(ft, fg):
        test_result, gold_result = json.loads(test_line), json.loads(gold_line)

        """ Test. """
        test_events, test_roles = get_event_role_list(test_result)
        event_pred += len(test_events)
        pred_number += len(test_roles)
        """ Gold. """
        gold_events, gold_roles = get_event_role_list(gold_result)
        event_gold += len(gold_events)
        gold_number += len(gold_roles)
        """ Compare. """
        for test_event in test_events:
            if test_event in gold_events: event_correct += 1
        for test_role in test_roles:
            if test_role in gold_roles: correct += 1
        """ Write results. """
        if parse_path is not None:
            result = {"pred_correct": [], "pred_wrong": [], "gold_lost": []}
            for test_role in test_roles:
                if test_role in gold_roles:
                    result["pred_correct"].append(test_role)
                else:
                    result["pred_wrong"].append(test_role)
            for gold_role in gold_roles:
                if gold_role not in test_roles:
                    result["gold_lost"].append(gold_role)
            fw.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    event_p = event_correct / event_pred if event_pred != 0 else 0
    event_r = event_correct / event_gold
    event_f1 = 2 * event_p * event_r / (event_p + event_r) if event_p != 0 else 0
    print("Event: total_right, total_predict, predict_right: {}, {}, {}".format(event_gold, event_pred, event_correct))
    print("Event: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(event_p, event_r, event_f1))

    p = correct / pred_number if pred_number != 0 else 0
    r = correct / gold_number
    f1 = 2 * p * r / (p + r) if p != 0 else 0
    print("Role: total_right, total_predict, predict_right: {}, {}, {}".format(gold_number, pred_number, correct))
    print("Role: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(p, r, f1))        

    ft.close()
    fg.close()
    if parse_path is not None: fw.close()

def get_location_pair(entity, sentence):
    start = sentence.find(entity)
    if start == -1: return None
    end = start + len(entity)
    return (start, end)

def evaluate_token_level(test_path, gold_path, parse_path):
    ft = open(test_path, "r", encoding="utf-8")
    fg = open(gold_path, "r", encoding="utf-8")

    correct, gold_number, pred_number = 0, 0, 0
    event_correct, event_gold, event_pred = 0, 0, 0

    for test_line, gold_line in zip(ft, fg):
        test_result, gold_result = json.loads(test_line), json.loads(gold_line)

        """ Test. """
        test_events, test_roles = get_event_role_list(test_result, "tuple")
        event_pred += len(test_events)
        for test_role in test_roles:
            pred_number += len(test_role[2])
        """ Gold. """
        gold_events, gold_roles = get_event_role_list(gold_result, "tuple")
        event_gold += len(gold_events)
        for gold_role in gold_roles:
            gold_number += len(gold_role[2])
        """ Compare. """
        for test_event in test_events:
            if test_event in gold_events: event_correct += 1
        for test_role in test_roles:
            cocurrent_token = 0
            for gold_role in gold_roles:
                if test_role[0] == gold_role[0] and test_role[1] == gold_role[1]:
                    test_pair = get_location_pair(test_role[2], test_result["text"])
                    gold_pair = get_location_pair(gold_role[2], gold_result["text"])
                    cross = min(test_pair[1], gold_pair[1]) - max(test_pair[0], gold_pair[0])
                    if cocurrent_token < cross: cocurrent_token = cross
            correct += cocurrent_token
    
    event_p = event_correct / event_pred if event_pred != 0 else 0
    event_r = event_correct / event_gold
    event_f1 = 2 * event_p * event_r / (event_p + event_r) if event_p != 0 else 0
    print("Event: total_right, total_predict, predict_right: {}, {}, {}".format(event_gold, event_pred, event_correct))
    print("Event: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(event_p, event_r, event_f1))

    p = correct / pred_number if pred_number != 0 else 0
    r = correct / gold_number
    f1 = 2 * p * r / (p + r) if p != 0 else 0
    print("Role: total_right, total_predict, predict_right: {}, {}, {}".format(gold_number, pred_number, correct))
    print("Role: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(p, r, f1))        

    ft.close()
    fg.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_path", default=None, type=str, required=True, help="Path of the predicted file.")
    parser.add_argument("--gold_path", default=None, type=str, required=True, help="Path of the gold file.")
    parser.add_argument("--parse_path", default=None, type=str, help="Path to write the parsing results.")
    parser.add_argument("--eval_type", choices=["strict", "token"],
        default="strict",
        help="What kind of eval function do you want to use.")
    args = parser.parse_args()
    if args.eval_type == "strict":
        evaluate(args.test_path, args.gold_path, args.parse_path)
    else:
        evaluate_token_level(args.test_path, args.gold_path, args.parse_path)