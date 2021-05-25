#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.25

import json
import argparse

def get_event_role_list(result):
    events, roles = set(), []
    for event in result["event_list"]:
        event_type = event["event_type"]
        events.add(event_type)
        for argument in event["arguments"]:
            roles.append(str((event_type, argument["role"], argument["argument"])))
    return list(events), roles

def evaluate(test_path, gold_path):
    ft = open(test_path, "r", encoding="utf-8")
    fg = open(gold_path, "r", encoding="utf-8")

    correct, gold_number, pred_number = 0, 0, 0
    event_correct, event_gold, event_pred = 0, 0, 0

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
    args = parser.parse_args()
    evaluate(args.test_path, args.gold_path)