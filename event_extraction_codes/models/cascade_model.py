#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.06.14

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(uer_dir)

import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from uer.layers import *
from uer.encoders import *
from event_extraction_codes.config import *

class CascadeModel(nn.Module):
    def __init__(self, args):
        super(CascadeModel, self).__init__()
        self.device = args.device
        self.events_num = args.events_num
        self.hidden_size = args.hidden_size
        self.entity_gate = args.entity_gate
        self.model_type = args.model_type
        self.event_role_len = len(args.schema_dict.event_role_list)

        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.triggers_head_output_layer = nn.Linear(self.hidden_size, 1)
        self.triggers_tail_output_layer = nn.Linear(self.hidden_size, 1)
        self.sigmoid_layer = nn.Sigmoid()
        self.arguments_head_output_layer = nn.Linear(self.hidden_size, self.event_role_len)
        self.arguments_tail_output_layer = nn.Linear(self.hidden_size, self.event_role_len)

        self.criterion = self.__init_criterion__()

    def __init_criterion__(self):
        if "bias" not in self.model_type:
            return BCEWithLogitsLoss(reduction="mean")
        else:
            return BCEWithLogitsLoss(reduction="mean", pos_weight=torch.Tensor([bias_weight]))

    def forward(self, batch, epoch):
        """ 
            Args:   
                batch:              [text, tokens, tokens_id, seg, triggers_head,
                                    triggers_tail, trigger_head, trigger_tail, arguments_head,
                                    arguments_tail, event_id, roles_list]      
            returns:
                feats:              (triggers_head_feats, triggers_tail_feats, arguments_head_feats, arguments_tail_feats)
        """
        # src:      tokens_id [batch_size x seq_length]
        # seg:      seg [batch_size x seq_length]
        src, seg = batch[2], batch[3]
        trigger_head, trigger_tail = batch[6], batch[7]
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        hidden = self.encoder(emb, seg)
        # Triggers head.
        triggers_head_output = self.triggers_head_output_layer(hidden).view(batch_size, seq_len)
        # Triggers tail.
        triggers_tail_output = self.triggers_tail_output_layer(hidden).view(batch_size, seq_len)
        # Get the trigger feature.
        batch_idxs = [_ for _ in range(batch_size)]
        head_feature = hidden[[batch_idxs, trigger_head]]
        tail_feature = hidden[[batch_idxs, trigger_tail]]
        trigger_feature = (head_feature + tail_feature) * 0.5
        hidden = hidden + trigger_feature.view(batch_size, 1, self.hidden_size)
        # Arguments_head.
        arguments_head_output = self.arguments_head_output_layer(hidden).view(batch_size, self.event_role_len, seq_len)
        # Arguments_tail.
        arguments_tail_output = self.arguments_tail_output_layer(hidden).view(batch_size, self.event_role_len, seq_len)
        return triggers_head_output, triggers_tail_output, arguments_head_output, arguments_tail_output
    
    def test(self, batch):
        text, src, seg = batch[0], batch[2], batch[3]
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        hidden = self.encoder(emb, seg)
        # Triggers head.
        triggers_head_output = self.triggers_head_output_layer(hidden).view(batch_size, seq_len)
        triggers_head_output = self.sigmoid_layer(triggers_head_output)
        # Triggers tail.
        triggers_tail_output = self.triggers_tail_output_layer(hidden).view(batch_size, seq_len)
        triggers_tail_output = self.sigmoid_layer(triggers_tail_output)
        # Get arguments.
        arguments_list = [set() for _ in range(batch_size)]
        for idx in range(batch_size):
            trigger_heads = np.where(triggers_head_output[idx].cpu() > self.entity_gate)[0]
            trigger_tails = np.where(triggers_tail_output[idx].cpu() > self.entity_gate)[0]
            triggers = []
            for trigger_head in trigger_heads:
                if trigger_head > len(text[idx]): continue              # Ignore [PAD].
                trigger_tail = trigger_tails[trigger_tails >= trigger_head]
                if len(trigger_tail) > 0:
                    trigger_tail = trigger_tail[0]
                    if trigger_tail > len(text[idx]): continue          # Ignore [PAD].
                    triggers.append((trigger_head, trigger_tail))
            for pair in triggers:
                # Get the trigger feature.
                trigger_feature = (hidden[idx][pair[0]] + hidden[idx][pair[1]]) * 0.5
                _hidden = hidden[idx] + trigger_feature
                # Arguments_head.
                arguments_head_output = self.arguments_head_output_layer(_hidden).view(self.event_role_len, seq_len)
                arguments_head_output = self.sigmoid_layer(arguments_head_output)
                # Arguments_tail.
                arguments_tail_output = self.arguments_tail_output_layer(_hidden).view(self.event_role_len, seq_len)
                arguments_tail_output = self.sigmoid_layer(arguments_tail_output)
                for event_role_idx in range(self.event_role_len):
                    argument_heads = np.where(arguments_head_output[event_role_idx].cpu() > self.entity_gate)[0]
                    argument_tails = np.where(arguments_tail_output[event_role_idx].cpu() > self.entity_gate)[0]
                    for argument_head in argument_heads:
                        if argument_head > len(text[idx]): continue     # Ignore [PAD].
                        argument_tail = argument_tails[argument_tails >= argument_head]
                        if len(argument_tail) > 0:
                            argument_tail = argument_tail[0]
                            if argument_tail > len(text[idx]): continue # Ignore [PAD].
                            arguments_list[idx].add((argument_head, argument_tail, event_role_idx))
        return arguments_list
   
    def get_batch(self, batch):
        text, tokens, tokens_id, seg, triggers_head, triggers_tail, trigger_head, \
            trigger_tail, arguments_head, arguments_tail, event_id, roles_list = batch
        tokens_id = tokens_id.to(self.device)
        seg = seg.to(self.device)
        triggers_head = triggers_head.to(self.device)
        triggers_tail = triggers_tail.to(self.device)
        arguments_head = arguments_head.to(self.device)
        arguments_tail = arguments_tail.to(self.device)
        return text, tokens, tokens_id, seg, triggers_head, triggers_tail, trigger_head, \
            trigger_tail, arguments_head, arguments_tail, event_id, roles_list
        
    def get_loss(self, feats, batch, epoch):
        """ 
            Args:
                feats:              (triggers_head_output, triggers_tail_output, arguments_head_output, arguments_tail_output)
                batch:              [text, tokens, tokens_id, seg, triggers_head,
                                    triggers_tail, trigger_head, trigger_tail, arguments_head,
                                    arguments_tail, event_id, roles_list]

            returns:
                loss
        """
        seg, gold_triggers_head, gold_triggers_tail = batch[3], batch[4], batch[5]
        gold_arguments_head, gold_arguments_tail = batch[8], batch[9]
        assert seg.shape == gold_triggers_head.shape
        batch_size, seq_len = seg.size(0), seg.size(1)
        # Mask the padding tags for gold.
        #print(gold_triggers_head.shape, seg.shape, gold_arguments_head.shape)
        masked_gold_triggers_head = torch.mul(gold_triggers_head, seg).float()
        masked_gold_triggers_tail = torch.mul(gold_triggers_tail, seg).float()
        masked_gold_arguments_head = torch.mul(gold_arguments_head, seg.view(batch_size, -1, seq_len)).float()
        masked_gold_arguments_tail = torch.mul(gold_arguments_tail, seg.view(batch_size, -1, seq_len)).float()
        masked_gold_feats = [masked_gold_triggers_head, masked_gold_triggers_tail, masked_gold_arguments_head, masked_gold_arguments_tail]

        # Mask the padding tags for pred.
        masked_pred_feats = []
        for idx in range(4):
            if idx < 2:
                masked_pred_feats.append(torch.mul(feats[idx], seg).float())
            else:
                masked_pred_feats.append(torch.mul(feats[idx], seg.view(batch_size, -1, seq_len)).float())
        
        # Calculate the loss.
        loss = 0
        for idx in range(4):
            loss += self.criterion(masked_pred_feats[idx].contiguous().view(-1), masked_gold_feats[idx].contiguous().view(-1))

        return loss

    def evaluate(self, args, batch, roles_list, is_test, f_write=None):
        text, tokens, gold_roles_list = batch[0], batch[1], batch[11]

        correct, gold_number, pred_number = 0, 0, 0
        event_correct, event_gold, event_pred = 0, 0, 0

        for idx in range(len(tokens)):
            correct += len(roles_list[idx] & gold_roles_list[idx])
            gold_number += len(gold_roles_list[idx])
            pred_number += len(roles_list[idx])
        
            event_list = set([item[2] for item in roles_list[idx]])
            gold_event_list = set([item[2] for item in gold_roles_list[idx]])
            event_correct += len(event_list & gold_event_list)
            event_gold += len(gold_event_list)
            event_pred += len(event_list)

            if not is_test: continue

            assert f_write is not None
            sentence = text[idx]
            pred_result = {"text": sentence, "event_list": []}
            event_dict = {}
            for role in roles_list[idx]:
                argument = sentence[role[0]: role[1] + 1]
                event_type, role_type = eval(args.schema_dict.event_role_list[role[2]])
                if event_type not in event_dict.keys():
                    event_dict[event_type] = []
                event_dict[event_type].append({"role": role_type, "argument": argument})
            for key, value in event_dict.items():
                pred_result["event_list"].append({"event_type": key, "arguments": value})        
            f_write.write(json.dumps(pred_result, ensure_ascii=False) + "\n")
        
        return correct, gold_number, pred_number, event_correct, event_gold, event_pred