#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

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

    def forward(self, src, seg):
        """ 
            Args:           
                src:                [batch_size x seq_length]
                seg:                [batch_size x seq_length]
            returns:
                feats:              [batch_size x events_num x seq_length x role_tags_number]
        """
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
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
        
    def get_loss(self, feats, gold_tags):
        """ 
            Args:
                feats:              [batch_size x events_num x seq_length x role_tags_number]
                gold_tags:          [batch_size x events_num x seq_length]
            returns:
                loss
        """
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

ModelDict = {
    "baseline": BaselineModel,
    "baseline-lstm": BaselineModel
}