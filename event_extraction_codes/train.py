#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

import json
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.seed import set_seed
from uer.opts import *
from event_extraction_codes.config import *
from event_extraction_codes.models import ModelDict
from event_extraction_codes.dataset import *
from event_extraction_codes.utils import check_file_path

# Model saver
def save_model_with_optim(model, optimizer, model_path):
    if hasattr(model, "module"):
        state_dict = {"net": model.module.state_dict(), "optimizer": optimizer.state_dict()}
    else:
        state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict()}
    check_file_path(model_path)
    torch.save(state_dict, model_path)

# k-fold file name.
def get_k_file_path(file_path, k_idx):
    file_name, extension = os.path.splitext(file_path)
    return file_name + str(k_idx) + extension

# Parameters loader.
def load_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=pretrained_model_path, type=str,
                        help="Path of the pretrained model file.")
    """ parser.add_argument("--last_model_path", default=last_model_path, type=str,
                        help="Path of the output last output model.")
    parser.add_argument("--best_model_path", default=best_model_path, type=str,
                        help="Path of the output best output model.")
    parser.add_argument("--result_path", default=result_path, type=str,
                        help="Path of the results.") """
    parser.add_argument("--middle_model_path", default=None, type=str,
                        help="Path of the middle model input path.")
    parser.add_argument("--save_dir_name", default="baseline", type=str,
                        help="Dir name for saving models and results")
    parser.add_argument("--vocab_path", default=vocabulary_path, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--config_path", default=config_path, type=str,
                        help="Path of the BERT config file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")

    # Model options.
    parser.add_argument("--model_type", 
        choices=["baseline", "baseline-lstm"],
        default="baseline",
        help="What kind of model do you want to use.")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=seq_length, type=int,
                        help="Sequence length.")
    model_opts(parser)
    
    # Training options.
    parser.add_argument("--epochs_num", type=int, default=epochs_num,
                        help="Number of epochs.")
    parser.add_argument("--eval_steps", type=int, default=eval_epoch,
                        help="Specific steps to eval the model.")
    parser.add_argument("--report_steps", type=int, default=report_steps,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=seed,
                        help="Random seed.")
    parser.add_argument("--K", type=int, default=K,
                        help="K fold.")
    
    # Optimization options.
    optimization_opts(parser)

    args = parser.parse_args()

    args = load_hyperparam(args)

    # Labels list.
    args.scheme_dict = SchemaDict()
    args.tokenizer = EventCharTokenizer(args)
    args.role_tags_num = len(role_label_list)
    args.events_num = args.scheme_dict.get_event_len()
    # Vocabulary.
    args.vocab = args.tokenizer.vocab
    args.vocab_len = len(args.vocab)
    # Scheduler
    #args.scheduler = scheduler
    args.power = power
    # Save path
    args.last_model_path = os.path.join(uer_dir, "result_models/" + args.save_dir_name + "/last/model.bin")
    args.best_model_path = os.path.join(uer_dir, "result_models/" + args.save_dir_name + "/best/model.bin")
    result_path = os.path.join(uer_dir, "results/" + args.save_dir_name + "/test_result.txt")

    if torch.cuda.is_available(): args.use_cuda = True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

# Model builder.
def build_model(args):

    # Build sequence labeling model.
    model = ModelDict[args.model_type](args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model = model.module

    # Load middle model.
    args.state_dict = {}
    if args.middle_model_path is not None:
        print("There is a middle model, let's use it!")
        args.state_dict = torch.load(args.middle_model_path)
        model.load_state_dict(args.state_dict["net"], strict=False)
    
    model = model.to(args.device)

    return model

def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    elif args.scheduler in ["polynomial"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps, power=args.power)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler

# Evaluation function.
def evaluate(model, args, is_test, k_idx=None, update_flag=True):
    if is_test:
        event_dataset = EventDataset(args, TEST, update=update_flag)
        # When evaluating the test set, the batch must be 1.
        event_data_loader = DataLoader(event_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        if k_idx is not None:
            result_file = get_k_file_path(args.result_path, k_idx)
        else:
            result_file = args.result_path
        check_file_path(result_file)
        fw = open(result_file, "w", encoding="utf-8")
    else:
        assert k_idx is not None
        event_dataset = EventDataset(args, VALID, k_idx, update=update_flag)
        event_data_loader = DataLoader(event_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    correct, gold_number, pred_number = 0, 0, 0
    event_correct, event_gold, event_pred = 0, 0, 0

    model.eval()

    for i, batch in enumerate(event_data_loader):

        text, tokens, tokens_id, tags, seg = batch
        tokens_id = tokens_id.to(args.device)
        seg = seg.to(args.device)

        feats = model(tokens_id, seg)

        best_path = model.get_best_path(feats)

        for j in range(0, len(tokens_id)):
            sentence = text[j]
            if is_test:
                pred_result = {"text": sentence, "event_list": []}
            for event_id in range(0, args.events_num):
                gold_tags = [str(role_label_list[int(p)]) for p in tags[j][event_id]]
                pred_tags = [str(role_label_list[p]) for p in best_path[j][event_id]]
                sentence_len = len(sentence)

                """ if i == 0 and j == 0:
                    print("sentence: ", sentence) """

                """ Evaluate. """
                for k in range(sentence_len):
                    # Gold.
                    if gold_tags[k][0] == "S" or gold_tags[k][0] == "B":
                        gold_number += 1
                    # Predict.
                    if pred_tags[k][0] == "S" or pred_tags[k][0] == "B":
                        pred_number += 1
                
                pred_pos, gold_pos = [], []
                start, end = 0, 0
                # Correct.
                for k in range(sentence_len):
                    if gold_tags[k][0] == "S":
                        start = k
                        end = k + 1
                    elif gold_tags[k][0] == "B":
                        start = k
                        end = k + 1
                        while end < sentence_len:
                            if gold_tags[end][0] == "I": end += 1
                            elif gold_tags[end][0] == "E":
                                end += 1
                                break
                            else: break
                    else:
                        continue
                    gold_pos.append((start, end))
                if len(gold_pos) > 0: event_gold += 1
                # Predict
                for k in range(sentence_len):
                    if pred_tags[k][0] == "S":
                        start = k
                        end = k + 1
                    elif pred_tags[k][0] == "B":
                        start = k
                        end = k + 1
                        while end < sentence_len:
                            if pred_tags[end][0] == "I": end += 1
                            elif pred_tags[end][0] == "E":
                                end += 1
                                break
                            else: break
                    else:
                        continue
                    pred_pos.append((start, end))
                if len(pred_pos) > 0: 
                    event_pred += 1
                    """ Get the results. """
                    if is_test:
                        event_type = args.scheme_dict.get_event_type(event_id)
                        pred_arguments = []
                        for pair in pred_pos:
                            role_type = args.scheme_dict.get_role_type(event_id, int(pred_tags[pair[0]].split("-")[-1]))
                            if role_type is not None:
                                pred_arguments.append({"role": role_type, "argument": sentence[pair[0]:pair[1]]})
                        pred_result["event_list"].append({"event_type": event_type, "arguments": pred_arguments})
                
                for pair in pred_pos:
                    if pair not in gold_pos: continue
                    if gold_tags[pair[0]] == pred_tags[pair[0]]:
                        correct += 1
                    """ for k in range(pair[0], pair[1]):
                        if gold_tags[k] != pred_tags[k]: 
                            break
                    else: 
                        correct += 1 """
                if len(pred_pos) > 0 and len(gold_pos) > 0: event_correct += 1
            
            if is_test:
                fw.write(json.dumps(pred_result, ensure_ascii=False) + "\n")

    if is_test:
        fw.close()

    event_p = event_correct / event_pred if event_pred != 0 else 0
    event_r = event_correct / event_gold
    event_f1 = 2 * event_p * event_r / (event_p + event_r) if event_p != 0 else 0
    print("Event: total_right, total_predict, predict_right: {}, {}, {}".format(event_gold, event_pred, event_correct))
    print("Event: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(event_p, event_r, event_f1))

    p = correct / pred_number if pred_number != 0 else 0
    r = correct / gold_number
    f1 = 2*p*r/(p+r) if p != 0 else 0
    print("Role: total_right, total_predict, predict_right: {}, {}, {}".format(gold_number, pred_number, correct))
    print("Role: precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(p, r, f1))

    return f1

# Training function.
# If args.K > 1, we apply K-fold validation.
# If args.K == 1, we use all of training data to train the best model.
def train_kfold(args):
    # Training phase.
    print("Start training.")
    
    for k_idx in range(args.K):
        total_loss, f1, best_f1 = 0., 0., 0.

        model = build_model(args)

        # Evaluate the middle model.
        if args.middle_model_path is not None and k_idx == 0:
            print("Start evaluate middle model.")
            best_f1 = evaluate(model, args, True, k_idx=0, update_flag=True)
            exit()

        print("--------------- The {}-th fold as the validation set... ---------------".format(k_idx+1))

        # Get the training data.
        update_flag = True if k_idx == 0 else False
        event_dataset = EventDataset(args, TRAIN, k_idx, update=update_flag)
        event_data_loader = DataLoader(event_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        args.train_steps = int(len(event_dataset) * args.epochs_num / args.batch_size) + 1

        # Optimizer.
        optimizer, scheduler = build_optimizer(args, model)
        """ if args.middle_model_path is not None:
            print("Loading optimizer...")
            optimizer.load_state_dict(args.state_dict["optimizer"]) """

        if k_idx == 0:
            print("Data length: ", len(event_dataset))

        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, batch in enumerate(event_data_loader):
                model.zero_grad()

                text, tokens, tokens_id, tags, seg = batch
                tokens_id = tokens_id.to(args.device)
                tags = tags.to(args.device)
                seg = seg.to(args.device)

                feats = model(tokens_id, seg)

                #print(feats.shape, tags.shape)
                loss = model.get_loss(feats, tags)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)

                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Loss: {:.6f}".format(epoch, i+1, loss))

                loss.backward()
                optimizer.step()
                scheduler.step()

            """ if epoch == 1:
                save_model_with_optim(model, optimizer, get_k_file_path(args.best_model_path, k_idx)) """

            if args.K > 1 and epoch % args.eval_steps == 0:
                f1 = evaluate(model, args, False, k_idx, update_flag=False)
                if f1 >= best_f1:
                    best_f1 = f1
                    save_model_with_optim(model, optimizer, get_k_file_path(args.best_model_path, k_idx))

        # Save the last optimizer and model.
        save_model_with_optim(model, optimizer, get_k_file_path(args.last_model_path, k_idx))

        # Evaluation phase.
        print("Start evaluation.")

        if args.K > 1:
            model.load_state_dict(torch.load(get_k_file_path(args.best_model_path, k_idx)), strict=False)
        else:
            model.load_state_dict(torch.load(get_k_file_path(args.last_model_path, k_idx)), strict=False)

        evaluate(model, args, True, k_idx, update_flag=update_flag)

def main():
    args = load_parameters()
    set_seed(args.seed)
    train_kfold(args)

if __name__ == "__main__":
    main()