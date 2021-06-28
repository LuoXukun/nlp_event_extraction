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

class HierarchicalModel(nn.Module):
    def __init__(self, args):
        super(HierarchicalModel, self).__init__()
        self.device = args.device
        self.events_num = args.events_num
        self.max_role_len = args.max_role_len
        self.hidden_size = args.hidden_size
        self.entity_gate = args.entity_gate
        self.role_gate = args.role_gate
        self.model_type = args.model_type

        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.entities_head_output_layer = nn.Linear(self.hidden_size, 1)
        self.entities_tail_output_layer = nn.Linear(self.hidden_size, 1)
        self.sigmoid_layer = nn.Sigmoid()
        self.entity_roles_output_layer = nn.Linear(self.hidden_size, self.events_num * self.max_role_len)

        self.criterion = self.__init_criterion__()

    def __init_criterion__(self):
        if "bias" not in self.model_type:
            return BCEWithLogitsLoss(reduction="mean")
        else:
            return BCEWithLogitsLoss(reduction="mean", pos_weight=torch.Tensor([bias_weight]))

    def forward(self, batch, epoch):
        """ 
            Args:   
                batch:              [text, tokens, tokens_id, seg, entities_head,
                                    entities_tail, entity_head, entity_tail, entity_roles, roles_list, events_id]        
            returns:
                feats:              (entities_head_feats, entities_tail_feats, role_feats)
        """
        # src:      tokens_id [batch_size x seq_length]
        # seg:      seg [batch_size x seq_length]
        src, seg = batch[2], batch[3]
        entity_head, entity_tail, events_id = batch[6], batch[7], batch[10]
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        hidden = self.encoder(emb, seg)
        # Entities head.
        entities_head_output = self.entities_head_output_layer(hidden).view(batch_size, seq_len)
        # Entities tail.
        entities_tail_output = self.entities_tail_output_layer(hidden).view(batch_size, seq_len)
        # Get the entity feature.
        batch_idxs = [_ for _ in range(batch_size)]
        head_feature = hidden[[batch_idxs, entity_head]]
        tail_feature = hidden[[batch_idxs, entity_tail]]
        entity_feature = (head_feature + tail_feature) * 0.5
        cls_hidden = hidden[:, 0, :] + entity_feature
        # Entities roles.
        entity_roles_output = self.entity_roles_output_layer(cls_hidden).view(batch_size, self.events_num, self.max_role_len)
        return entities_head_output, entities_tail_output, entity_roles_output
    
    def test(self, batch):
        text, src, seg = batch[0], batch[2], batch[3]
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        hidden = self.encoder(emb, seg)
        # Entities head.
        entities_head_output = self.entities_head_output_layer(hidden).view(batch_size, seq_len)
        entities_head_output = self.sigmoid_layer(entities_head_output)
        # Entities tail.
        entities_tail_output = self.entities_tail_output_layer(hidden).view(batch_size, seq_len)
        entities_tail_output = self.sigmoid_layer(entities_tail_output)
        # Get heads and tails.
        roles_list = [set() for _ in range(batch_size)]
        for idx in range(batch_size):
            entity_heads = np.where(entities_head_output[idx].cpu() > self.entity_gate)[0]
            entity_tails = np.where(entities_tail_output[idx].cpu() > self.entity_gate)[0]
            entities = []
            for entity_head in entity_heads:
                if entity_head == 0: continue                       # Ignore [CLS].
                if entity_head > len(text[idx]): continue           # Ignore [PAD].
                entity_tail = entity_tails[entity_tails >= entity_head]
                if len(entity_tail) > 0:
                    entity_tail = entity_tail[0]
                    if entity_tail > len(text[idx]): continue       # Ignore [PAD].
                    entities.append((entity_head, entity_tail))
            for pair in entities:
                # Get the entity feature.
                entity_feature = (hidden[idx][pair[0]] + hidden[idx][pair[1]]) * 0.5
                cls_hidden = hidden[idx][0] + entity_feature
                # Entities roles.
                entity_roles_output = self.entity_roles_output_layer(cls_hidden)
                entity_roles_output = self.sigmoid_layer(entity_roles_output).view(self.events_num, self.max_role_len)
                event_ids, role_ids = np.where(entity_roles_output.cpu() > self.role_gate)
                if len(event_ids) == 0: continue
                for event_id, role_id in zip(event_ids, role_ids):
                    roles_list[idx].add((pair[0], pair[1], event_id, role_id))
        return roles_list
   
    def get_batch(self, batch):
        text, tokens, tokens_id, seg, entities_head, entities_tail, entity_head, entity_tail, entity_roles, roles_list = batch
        tokens_id = tokens_id.to(self.device)
        seg = seg.to(self.device)
        entity_roles = entity_roles.to(self.device)
        entities_head = entities_head.to(self.device)
        entities_tail = entities_tail.to(self.device)
        return text, tokens, tokens_id, seg, entities_head, entities_tail, entity_head, entity_tail, entity_roles, roles_list
        
    def get_loss(self, feats, batch, epoch):
        """ 
            Args:
                feats:              (entities_head_output, entities_tail_output, entity_roles_output)
                batch:              [text, tokens, tokens_id, seg, entities_head, \
                                    entities_tail, entity_head, entity_tail, entity_roles, roles_list]

            returns:
                loss
        """
        seg, gold_entities_head, gold_entities_tail, gold_roles = batch[3][:, 1:], batch[4][:, 1:], batch[5][:, 1:], batch[8]
        assert seg.shape == gold_entities_head.shape
        # Mask the padding tags for gold.
        masked_gold_entities_head = torch.mul(gold_entities_head, seg).float()
        masked_gold_entities_tail = torch.mul(gold_entities_tail, seg).float()

        # Mask the padding tags for pred.
        head_feats, tail_feats = feats[0][:, 1:], feats[1][:, 1:]
        head_feats = torch.mul(head_feats, seg).float()
        tail_feats = torch.mul(tail_feats, seg).float()

        role_feats = feats[2].float()
        gold_roles = gold_roles.float()

        head_loss = self.criterion(head_feats.contiguous().view(-1), masked_gold_entities_head.contiguous().view(-1))
        tail_loss = self.criterion(tail_feats.contiguous().view(-1), masked_gold_entities_tail.contiguous().view(-1))
        role_loss = self.criterion(role_feats.contiguous().view(-1), gold_roles.contiguous().view(-1))
        if "bias" not in self.model_type:
            return head_loss + tail_loss + role_loss
        else:
            seq_len = float(seg.size(1))        # Only estimating.
            role_len = float(self.events_num * self.max_role_len)
            return (role_len / seq_len) * (head_loss + tail_loss) + role_loss

    def evaluate(self, args, batch, roles_list, is_test, f_write=None):
        text, tokens, gold_roles_list = batch[0], batch[1], batch[9]

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
                argument = sentence[role[0] - 1: role[1]]
                event_type = args.schema_dict.get_event_type(role[2])
                assert event_type is not None
                role_type = args.schema_dict.get_role_type(role[2], role[3])
                if role_type is None: continue
                if event_type not in event_dict.keys():
                    event_dict[event_type] = []
                event_dict[event_type].append({"role": role_type, "argument": argument})
            for key, value in event_dict.items():
                pred_result["event_list"].append({"event_type": key, "arguments": value})        
            f_write.write(json.dumps(pred_result, ensure_ascii=False) + "\n")
        
        return correct, gold_number, pred_number, event_correct, event_gold, event_pred