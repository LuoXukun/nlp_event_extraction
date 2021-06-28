#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(uer_dir)

import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from uer.layers import *
from uer.encoders import *
from event_extraction_codes.config import *

class BaselineModel(nn.Module):
    def __init__(self, args):
        super(BaselineModel, self).__init__()
        self.device = args.device
        self.events_num = args.events_num
        self.role_tags_num = args.role_tags_num
        self.use_lstm_decoder = True if args.model_type == "baseline-lstm" else False
        self.hidden_size = args.hidden_size

        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        if self.use_lstm_decoder:
            self.decoder = nn.LSTM(input_size=self.hidden_size,
                           hidden_size=self.hidden_size // 2,
                           num_layers=2,
                           bidirectional=True,
                           dropout=args.dropout,
                           batch_first=True)
            self.droplayer = nn.Dropout(p=args.dropout)
        self.output_layer = nn.Linear(self.hidden_size, self.events_num * self.role_tags_num)
        self.criterion = self.__init_criterion__()

    def __init_criterion__(self):
        weight = [baseline_criterion_weigth[-1]] * self.role_tags_num
        weight[LABEL_O] = baseline_criterion_weigth[LABEL_O]
        weight[LABEL_P] = baseline_criterion_weigth[LABEL_P]
        weight = torch.tensor(weight).to(self.device)
        print("weight: ", weight)
        return CrossEntropyLoss(weight, reduction="sum")

    def __train__(self, batch):
        """ 
            Args:   
                batch:              [text, tokens, tokens_id, seg, tags]        
            returns:
                feats:              [batch_size x events_num x seq_length x role_tags_number]
        """
        # src:      tokens_id [batch_size x seq_length]
        # seg:      seg [batch_size x seq_length]
        src, seg = batch[2], batch[3]
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        #print(src.shape, seg.shape)
        emb = self.embedding(src, seg)
        # Encoder.
        hidden = self.encoder(emb, seg)
        # Decoder.
        if self.use_lstm_decoder:
            lstm_out, _ = self.decoder(hidden)
            hidden = lstm_out.contiguous().view(-1, self.hidden_size)
            hidden = self.droplayer(hidden)
        # Feats.
        # [batch_size x seq_length x events_num x role_tags_number]
        feats = self.output_layer(hidden).view(batch_size, seq_len, self.events_num, self.role_tags_num)
        # [batch_size x events_num x seq_length x role_tags_number]
        feats = torch.transpose(feats, 1, 2)
        return feats
    
    def forward(self, batch, epoch):
        return self.__train__(batch)

    def test(self, batch):
        return self.__train__(batch)
    
    def get_batch(self, batch):
        text, tokens, tokens_id, seg, tags = batch
        tokens_id = tokens_id.to(self.device)
        tags = tags.to(self.device)
        seg = seg.to(self.device)
        return text, tokens, tokens_id, seg, tags
        
    def get_loss(self, feats, batch, epoch):
        """ 
            Args:
                feats:              [batch_size x events_num x seq_length x role_tags_number]
                batch:              [text, tokens, tokens_id, seg, tags] 
            returns:
                loss
        """
        # gold_tags:        tags [batch_size x events_num x seq_length]
        gold_tags = batch[4]
        seq_len = feats.size(2)
        return self.criterion(feats.contiguous().view(-1, self.role_tags_num), gold_tags.contiguous().view(-1)) / seq_len
    
    def get_best_path(self, feats):
        """ 
            Args:
                feats:              [batch_size x events_num x seq_length x role_tags_number]
            returns:
                best_path:          [batch_size x events_num x seq_length]
        """
        batch_size, events_num = feats.size(0), feats.size(1)
        return torch.argmax(feats, dim=-1).view(batch_size, events_num, -1)

    def evaluate(self, args, batch, feats, is_test, f_write=None):
        text, tokens, tokens_id, seg, tags = batch
        best_path = self.get_best_path(feats)

        correct, gold_number, pred_number = 0, 0, 0
        event_correct, event_gold, event_pred = 0, 0, 0

        for j in range(0, len(tokens_id)):
            sentence = text[j]
            if is_test:
                pred_result = {"text": sentence, "event_list": []}
            for event_id in range(0, args.events_num):
                gold_tags = [str(role_label_list[int(p)]) for p in tags[j][event_id]]
                pred_tags = [str(role_label_list[p]) for p in best_path[j][event_id]]
                sentence_len = len(sentence)

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
                        event_type = args.schema_dict.get_event_type(event_id)
                        pred_arguments = []
                        for pair in pred_pos:
                            role_type = args.schema_dict.get_role_type(event_id, int(pred_tags[pair[0]].split("-")[-1]))
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
                assert f_write is not None
                f_write.write(json.dumps(pred_result, ensure_ascii=False) + "\n")
        
        return correct, gold_number, pred_number, event_correct, event_gold, event_pred
