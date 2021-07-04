#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

import os
import json
import torch
import argparse
import numpy as np
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from event_extraction_codes.config import *
from event_extraction_codes.utils import *
from uer.utils.vocab import Vocab
from uer.utils.constants import CLS_TOKEN, PAD_ID, WHITE_SPACE_TOKEN, PUNC_TOKEN
from uer.utils.tokenizers import Tokenizer, convert_tokens_to_ids, _is_whitespace, _is_punctuation

class EventCharTokenizer(Tokenizer):
        
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        tokens = list(text.lower())
        for i in range(len(tokens)):
            if _is_whitespace(tokens[i]):
                tokens[i] = WHITE_SPACE_TOKEN
                continue
            if _is_punctuation(tokens[i]):
                tokens[i] = PUNC_TOKEN
        if use_vocab:
            return [token if token in self.vocab else "[UNK]" for token in tokens]
        else:
            return tokens

class SchemaDict():
    def __init__(self):
        self.schema_path = origin_schema_path
        self.max_role_len = 6
        self.event_dict, self.event_list = {}, []
        self.__init_event__()
        self.__init_role__()
        self.__init_event_role__()
        
    def __init_event__(self):
        with open(self.schema_path, "r", encoding="utf-8") as fr:
            for line_id, line in enumerate(fr):
                line = line.strip()
                if line == "": continue
                line = json.loads(line)
                self.event_dict[line["event_type"]] = len(self.event_list)
                self.event_list.append(line["event_type"])
    
    def __init_role__(self):
        self.role_dict = [{} for _ in range(len(self.event_list))]
        self.role_list = [[] for _ in range(len(self.event_list))]
        with open(self.schema_path, "r", encoding="utf-8") as fr:
            for line_id, line in enumerate(fr):
                line = line.strip()
                if line == "": continue
                line = json.loads(line)["role_list"]
                for item in line:
                    self.role_dict[line_id][item["role"]] = len(self.role_list[line_id])
                    self.role_list[line_id].append(item["role"])

    def __init_event_role__(self):
        self.event_role_list, self.event_role_dict = [], {}
        for idx, event in enumerate(self.event_list):
            for role_type in self.role_list[idx]:
                self.event_role_dict[str((event, role_type))] = len(self.event_role_list)
                self.event_role_list.append(str((event, role_type)))
        #print("event_role_list: {}\nevent_role_dict: {}".format(self.event_role_list, json.dumps(self.event_role_dict, ensure_ascii=False)))
    
    def get_event_id(self, event_type):
            return self.event_dict.get(event_type, None)
    
    def get_event_type(self, event_index):
        try:
            return self.event_list[event_index]
        except:
            return None
    
    def get_event_len(self):
        return len(self.event_list)
    
    def get_role_id(self, event_id, role_type):
            return self.role_dict[event_id].get(role_type, None)

    def get_role_type(self, event_id, role_index):
        try:
            return self.role_list[event_id][role_index]
        except:
            return None
        

class EventDataset(Dataset):
    def __init__(self, args, state=TRAIN, index=0, update=False):
        """ 
            The Event Dataset.

            Args:

                args:           Some arguments.
                train:          If the training set, the validation set or the test set. 
                                0->TRAIN, 1->VALID, 2->TEST.
                index:          The i-th fold in the dataset.
                update:         To update the json file.

            Returns:
        """
        self.state = state
        self.index = index
        self.k_fold = args.K
        self.vocab_path = args.vocab_path
        self.model_type = args.model_type

        self.origin_train_path = origin_train_path
        self.origin_test_path = origin_test_path
        self.preprocessed_train_path = preprocessed_train_path
        self.preprocessed_test_path = preprocessed_test_path

        self.origin_data_path = self.origin_test_path if state == TEST else self.origin_train_path
        self.preprocessed_data_path = self.preprocessed_test_path if state == TEST else self.preprocessed_train_path

        self.max_length, self.overlap_num = 0, 0
        self.schema_dict = SchemaDict()
        self.role_label_list = role_label_list
        self.role_label_dict = {self.role_label_list[_]:_ for _ in range(len(self.role_label_list))}

        # Tokenizer
        self.tokenizer = EventCharTokenizer(args)

        # Preprocess
        self.__preprocess_data__ = {
            "baseline": self.__preprocess_data_baseline__,
            "baseline_lstm": self.__preprocess_data_baseline__,
            "hierarchical": self.__preprocess_data_hierarchical__,
            "hierarchical-bias": self.__preprocess_data_hierarchical__,
            "cascade": self.__preprocess_data_cascade__,
            "cascade-bias": self.__preprocess_data_cascade__,
            "cascade-sample": self.__preprocess_data_cascade__
        }
        if not os.path.exists(self.preprocessed_data_path) or update is True:
            self.__preprocess_data__[self.model_type]()
        
        self.__get_data__ = {
            "baseline": self.__get_data_baseline__,
            "baseline_lstm": self.__get_data_baseline__,
            "hierarchical": self.__get_data_hierarchical__,
            "hierarchical-bias": self.__get_data_hierarchical__,
            "cascade": self.__get_data_cascade__,
            "cascade-bias": self.__get_data_cascade__,
            "cascade-sample": self.__get_data_cascade__
        }
        self.__get_data__[self.model_type]()
    
    """ Baseline: Normal sequence tagging. """
    def __preprocess_data_baseline__(self):
        """ Generate the preprocessed data. """
        print("Baseline: Preprocessing dataset...")
        check_file_path(self.preprocessed_data_path)
        fw = open(self.preprocessed_data_path, "w", encoding="utf-8")

        with open(self.origin_data_path, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                line = json.loads(line)
                """ 
                    {
                        "text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", 
                        "id": "cba11b5059495e635b4f95e7484b2684", 
                        "event_list": [
                            {
                                "event_type": "组织关系-裁员", 
                                "trigger": "裁员", 
                                "trigger_start_index": 15, 
                                "arguments": [
                                    {"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, 
                                    {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}
                                ], 
                                "class": "组织关系"
                            }
                        ]
                    } 
                """
                text, event_list = line["text"], line["event_list"]
                tokens = self.tokenizer.tokenize(line["text"])
                try:
                    assert len(text) == len(tokens)
                except:
                    print("Error when comparing text with tokens!")
                    print("text_len: {}, text: {}".format(len(text), text))
                    print("tokens_len: {}, tokens: {}".format(len(tokens), tokens))
                    exit()
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                text_len = len(text)
                # is_overlap = False
                if self.max_length < text_len:
                    self.max_length = text_len
                tags = [[LABEL_O for i in range(text_len)] for j in range(self.schema_dict.get_event_len())]
                for event in event_list:
                    event_type = event["event_type"]
                    event_type_id = self.schema_dict.get_event_id(event_type)
                    assert event_type_id is not None
                    arguments = event["arguments"]
                    for argument in arguments:
                        argument_start_index = int(argument["argument_start_index"])
                        argument_end_index = argument_start_index + len(argument["argument"])
                        role_label = self.schema_dict.get_role_id(event_type_id, argument["role"])

                        # Calculating the overlap number across event. Train: 1492.
                        """ if np.any(np.array(tags)[:, argument_start_index:argument_end_index]):
                            #print(np.array(tags)[:, argument_start_index:argument_end_index])
                            is_overlap = True """
                        
                        # Calculating the overlap number in event. Train: 841.
                        """ if np.any(np.array(tags)[event_type_id, argument_start_index:argument_end_index]):
                            is_overlap = True """
                        
                        if not np.any(np.array(tags)[:, argument_start_index:argument_end_index]):
                            self.__sequence_tag__(argument_start_index, argument_end_index, str(role_label), tags[event_type_id])
                preprocessed_sample = json.dumps({"text": text, "tokens": tokens, "tokens_id": tokens_id, "tags": tags}, ensure_ascii=False)
                # if is_overlap is True: self.overlap_num += 1
                # if idx == 0: print(preprocessed_sample)
                fw.write(preprocessed_sample + "\n")

        #print("Max length: {}, overlap_num: {}.".format(self.max_length, self.overlap_num))
        fw.close()

    def __get_data_baseline__(self):
        fr_pre = open(self.preprocessed_data_path, "r", encoding="utf-8")
        data_lines = fr_pre.readlines()
        data_len = len(data_lines)
        state, k_fold, index = self.state, self.k_fold, self.index

        if state == TRAIN:
            if k_fold > 1:
                self.data_lines = data_lines[:int((index % k_fold) * data_len / k_fold)] + \
                                    data_lines[int((index % k_fold + 1) * data_len / k_fold):]
            else:
                self.data_lines = data_lines
        elif state == VALID:
            if k_fold > 1:
                self.data_lines = data_lines[int((index % k_fold) * data_len / k_fold): \
                                    int((index % k_fold + 1) * data_len / k_fold)]
            else:
                self.data_lines = []
        else:
            self.data_lines = data_lines

        fr_pre.close()

        # Get the tokens and labels.
        self.text, self.tokens, self.tokens_id, self.tags = [], [], [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            self.text.append(line["text"])
            self.tokens.append(line["tokens"])
            self.tokens_id.append(line["tokens_id"])
            self.tags.append(line["tags"])
        self.length = len(self.text)
        
        return
    
    """ Hierarchical: Get the entities and then identify their role one by one. """
    def __preprocess_data_hierarchical__(self):
        print("Hierarchical: Preprocessing dataset...")
        check_file_path(self.preprocessed_data_path)
        fw = open(self.preprocessed_data_path, "w", encoding="utf-8")

        with open(self.origin_data_path, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                line = json.loads(line)
                text, event_list = line["text"], line["event_list"]
                tokens = [CLS_TOKEN] + self.tokenizer.tokenize(line["text"])
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                text_len = len(text) + 1

                # Get the dict like {entity_index: {(event_id, role_id),...}, ...}
                entities_dict = {}
                entities_head, entities_tail = [0 for i in range(text_len)], [0 for i in range(text_len)]
                for event in event_list:
                    event_type, arguments = event["event_type"], event["arguments"]
                    event_type_id = self.schema_dict.get_event_id(event_type)
                    assert event_type_id is not None
                    for argument in arguments:
                        argument_start_index = int(argument["argument_start_index"]) + 1
                        argument_end_index = argument_start_index + len(argument["argument"]) - 1
                        role_id = self.schema_dict.get_role_id(event_type_id, argument["role"])
                        entity_pair = (argument_start_index, argument_end_index)
                        if str(entity_pair) not in entities_dict.keys():
                            entities_dict[str(entity_pair)] = set()
                        entities_dict[str(entity_pair)].add((event_type_id, role_id))
                        entities_head[argument_start_index] = 1
                        entities_tail[argument_end_index] = 1
                for key in entities_dict.keys():
                    entities_dict[key] = list(entities_dict[key])

                # Generate the data.
                preprocessed_sample = json.dumps({
                    "text": text, "tokens": tokens, "tokens_id": tokens_id, 
                    "entities_head": entities_head, "entities_tail": entities_tail, "roles_dict": entities_dict
                }, ensure_ascii=False)
                fw.write(preprocessed_sample + "\n")
        fw.close()
    
    def __get_data_hierarchical__(self):
        fr_pre = open(self.preprocessed_data_path, "r", encoding="utf-8")
        data_lines = fr_pre.readlines()
        data_len = len(data_lines)
        state, k_fold, index = self.state, self.k_fold, self.index

        if state == TRAIN:
            if k_fold > 1:
                self.data_lines = data_lines[:int((index % k_fold) * data_len / k_fold)] + \
                                    data_lines[int((index % k_fold + 1) * data_len / k_fold):]
            else:
                self.data_lines = data_lines
        elif state == VALID:
            if k_fold > 1:
                self.data_lines = data_lines[int((index % k_fold) * data_len / k_fold): \
                                    int((index % k_fold + 1) * data_len / k_fold)]
            else:
                self.data_lines = []
        else:
            self.data_lines = data_lines

        fr_pre.close()

        self.text, self.tokens, self.tokens_id = [], [], []
        self.entities_head, self.entities_tail, self.roles_list = [], [], []
        self.entity_head, self.entity_tail, self.entity_roles = [], [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            # roles_dict = {(index1, index2): {(event_type_id, role_type_id), ...}}
            roles_dict = line["roles_dict"]
            if state != TRAIN:
                self.text.append(line["text"])
                self.tokens.append(line["tokens"])
                self.tokens_id.append(line["tokens_id"])
                self.entities_head.append(line["entities_head"])
                self.entities_tail.append(line["entities_tail"])
                self.entity_head.append(-1)     # Ignore it.
                self.entity_tail.append(-1)     # Ignore it.
                self.entity_roles.append([-1])    # Ignore it.
                roles_list = set()
                for key, value in roles_dict.items():
                    _key = eval(key)
                    for pair in value:
                        # (index1, index2, event_type_id, role_type_id)
                        roles_list.add((int(_key[0]), int(_key[1]), pair[0], pair[1]))
                self.roles_list.append(roles_list)
            else:
                for key, value in roles_dict.items():
                    _key = eval(key)
                    if len(list(value)) == 0: continue
                    self.text.append(line["text"])
                    self.tokens.append(line["tokens"])
                    self.tokens_id.append(line["tokens_id"])
                    self.entities_head.append(line["entities_head"])
                    self.entities_tail.append(line["entities_tail"])
                    self.roles_list.append(-1)  # Ignore it.
                    self.entity_head.append(_key[0])
                    self.entity_tail.append(_key[1])
                    entity_roles = [[0 for i in range(self.schema_dict.max_role_len)] for j in range(self.schema_dict.get_event_len())]
                    for pair in value:
                        entity_roles[pair[0]][pair[1]] = 1
                    self.entity_roles.append(entity_roles)
        #print("len: ", len(self.text), len(self.tokens), len(self.tokens_id), len(self.entities_head), \
        #    len(self.entities_tail), len(self.entity_head), len(self.entity_tail), len(self.entity_roles), len(self.roles_list))
        self.length = len(self.text)

    """ Cascade_Binary: Get the trigger and then the entities. """
    def __preprocess_data_cascade__(self):
        print("Cascade_Binary: Preprocessing dataset...")
        check_file_path(self.preprocessed_data_path)
        fw = open(self.preprocessed_data_path, "w", encoding="utf-8")

        with open(self.origin_data_path, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                line = json.loads(line)
                """ 
                    {
                        "text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", 
                        "id": "cba11b5059495e635b4f95e7484b2684", 
                        "event_list": [
                            {
                                "event_type": "组织关系-裁员", 
                                "trigger": "裁员", 
                                "trigger_start_index": 15, 
                                "arguments": [
                                    {"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, 
                                    {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}
                                ], 
                                "class": "组织关系"
                            }
                        ]
                    } 
                """
                text, event_list = line["text"], line["event_list"]
                tokens = self.tokenizer.tokenize(line["text"])
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                text_len = len(text)

                # Get the dict like {trigger_index: {"event_type": "", "arguments": {(start, end, event_role_id), ...}, ...}}
                trigger_dict = {}
                triggers_head, triggers_tail = [0 for i in range(text_len)], [0 for i in range(text_len)]
                for event in event_list:
                    event_type, arguments = event["event_type"], event["arguments"]
                    trigger_start_index = int(event["trigger_start_index"])
                    trigger_end_index = trigger_start_index + len(event["trigger"]) - 1
                    triggers_head[trigger_start_index] = 1
                    triggers_tail[trigger_end_index] = 1
                    trigger_pair = str((trigger_start_index, trigger_end_index))
                    if trigger_pair not in trigger_dict: 
                        trigger_dict[trigger_pair] = {"event_type": event_type, "arguments": []}
                    for argument in arguments:
                        argument_start_index = int(argument["argument_start_index"])
                        argument_end_index = argument_start_index + len(argument["argument"]) - 1
                        role_type = argument["role"]
                        event_role_id = self.schema_dict.event_role_dict[str((event_type, role_type))]
                        trigger_dict[trigger_pair]["arguments"].append((argument_start_index, argument_end_index, event_role_id))
                
                # Generate the data.
                preprocessed_sample = json.dumps({
                    "text": text, "tokens": tokens, "tokens_id": tokens_id,
                    "triggers_head": triggers_head, "triggers_tail": triggers_tail, "trigger_dict": trigger_dict
                }, ensure_ascii=False)
                fw.write(preprocessed_sample + "\n")
        fw.close()

    def __get_data_cascade__(self):
        fr_pre = open(self.preprocessed_data_path, "r", encoding="utf-8")
        data_lines = fr_pre.readlines()
        data_len = len(data_lines)
        state, k_fold, index = self.state, self.k_fold, self.index

        if state == TRAIN:
            if k_fold > 1:
                self.data_lines = data_lines[:int((index % k_fold) * data_len / k_fold)] + \
                                    data_lines[int((index % k_fold + 1) * data_len / k_fold):]
            else:
                self.data_lines = data_lines
        elif state == VALID:
            if k_fold > 1:
                self.data_lines = data_lines[int((index % k_fold) * data_len / k_fold): \
                                    int((index % k_fold + 1) * data_len / k_fold)]
            else:
                self.data_lines = []
        else:
            self.data_lines = data_lines

        fr_pre.close()
        self.text, self.tokens, self.tokens_id = [], [], []
        self.triggers_head, self.triggers_tail, self.roles_list = [], [], []
        self.trigger_head, self.trigger_tail, self.event_id = [], [], []
        self.arguments_head, self.arguments_tail = [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            # trigger_dict = {(trigger_start, trigger_end): {"event_type": "", "arguments": [(start, end, id), ...]}}
            trigger_dict = line["trigger_dict"]
            if state != TRAIN:
                self.text.append(line["text"])
                self.tokens.append(line["tokens"])
                self.tokens_id.append(line["tokens_id"])
                self.triggers_head.append(line["triggers_head"])
                self.triggers_tail.append(line["triggers_tail"])
                self.trigger_head.append(-1)        # Ignore it.
                self.trigger_tail.append(-1)        # Ignore it.
                self.event_id.append(-1)            # Ignore it.
                self.arguments_head.append([[-1]])  # Ignore it.
                self.arguments_tail.append([[-1]])  # Ignore it.
                roles_list = set()
                for key, value in trigger_dict.items():
                    for triple in value["arguments"]:
                        roles_list.add((triple[0], triple[1], triple[2]))
                self.roles_list.append(roles_list)
            else:
                for key, value in trigger_dict.items():
                    _key = eval(key)
                    self.text.append(line["text"])
                    self.tokens.append(line["tokens"])
                    self.tokens_id.append(line["tokens_id"])
                    self.triggers_head.append(line["triggers_head"])
                    self.triggers_tail.append(line["triggers_tail"])
                    self.roles_list.append(-1)      # Ignore it.
                    self.trigger_head.append(_key[0])
                    self.trigger_tail.append(_key[1])
                    self.event_id.append(self.schema_dict.get_event_id(value["event_type"]))
                    arguments_head = [[0 for i in range(len(line["tokens"]))] for j in range(len(self.schema_dict.event_role_list))]
                    arguments_tail = [[0 for i in range(len(line["tokens"]))] for j in range(len(self.schema_dict.event_role_list))]
                    for triple in value["arguments"]:
                        arguments_head[triple[2]][triple[0]] = 1
                        arguments_tail[triple[2]][triple[1]] = 1
                    self.arguments_head.append(arguments_head)
                    self.arguments_tail.append(arguments_tail)
        self.length = len(self.text)

    def __getitem__(self, index):
        if "baseline" in self.model_type:
            return self.text[index], self.tokens[index], self.tokens_id[index], self.tags[index]
        elif "hierarchical" in self.model_type:
            return self.text[index], self.tokens[index], self.tokens_id[index], self.entities_head[index], \
                self.entities_tail[index], self.entity_head[index], self.entity_tail[index], self.entity_roles[index], self.roles_list[index]
        elif "cascade" in self.model_type:
            return self.text[index], self.tokens[index], self.tokens_id[index], self.triggers_head[index], \
                self.triggers_tail[index], self.trigger_head[index], self.trigger_tail[index], self.arguments_head[index], \
                self.arguments_tail[index], self.event_id[index], self.roles_list[index]
        else:
            print("Undefined model type!")
            exit()

    def __len__(self):
        return self.length
    
    def __sequence_tag__(self, start, end, label, sequence):
        assert label is not None
        assert end <= len(sequence)
        assert end >= start
        if end == start: return

        if end - start == 1: # S
            sequence[start] = self.role_label_dict.get("S-" + label)
        else:
            sequence[start: end] = [self.role_label_dict.get("B-" + label)] + \
                                    [self.role_label_dict.get("I-" + label)] * (end - start - 2) + \
                                    [self.role_label_dict.get("E-" + label)]
        
        return

def collate_fn(batch, max_len=512, model_type="baseline"):
    """ 
        Collate_fn function. Arrange the batch in reverse order of length.

        Args:

            batch:              The batch data.
            max_len:            The max length.
            model_type:         The model type.
        
        Returns:
            ([text], [tokens], [tokens_id], [tags], [seg])
    """
    data_length = [len(b[1]) for b in batch]
    max_length = min(max(data_length), max_len)
    text, tokens = [b[0] for b in batch], [b[1] for b in batch]
    tokens_id = [torch.FloatTensor(b[2]) for b in batch]
    seg = [torch.FloatTensor([1] * _) for _ in data_length]
    tokens_id = pad_sequence(tokens_id, batch_first=True, padding_value=PAD_ID).long()
    seg = pad_sequence(seg, batch_first=True, padding_value=0).long()

    if model_type in ["baseline", "baseline-lstm"]:
        tags = [torch.FloatTensor(b[3]).T for b in batch]
        #print(tags[0].shape)
        tags = torch.transpose(pad_sequence(tags, batch_first=True, padding_value=LABEL_P).long(), 1, 2)
        #print(tags.shape)
        return text, tokens, tokens_id[:, :max_length], seg[:, :max_length], tags[:, :, :max_length]
    elif model_type in ["hierarchical", "hierarchical-bias"]:
        entities_head = [torch.FloatTensor(b[3]) for b in batch]
        entities_tail = [torch.FloatTensor(b[4]) for b in batch]
        entity_head = [b[5] for b in batch]
        entity_tail = [b[6] for b in batch]
        entity_roles = torch.LongTensor([b[7] for b in batch])
        entities_head = pad_sequence(entities_head, batch_first=True, padding_value=0).long()
        entities_tail = pad_sequence(entities_tail, batch_first=True, padding_value=0).long()
        roles_list = [b[8] for b in batch]
        return text, tokens, tokens_id[:, :max_length], seg[:, :max_length], entities_head[:, :max_length], \
            entities_tail[:, :max_length], entity_head, entity_tail, entity_roles, roles_list
    elif model_type in ["cascade", "cascade-bias", "cascade-sample"]:
        triggers_head = [torch.FloatTensor(b[3]) for b in batch]
        triggers_tail = [torch.FloatTensor(b[4]) for b in batch]
        trigger_head = [b[5] for b in batch]
        trigger_tail = [b[6] for b in batch]
        arguments_head = [torch.FloatTensor(c) for b in batch for c in b[7]]
        #print("length for arguemnts head: ", len(arguments_head))
        #print(arguments_head)
        arguments_tail = [torch.FloatTensor(c) for b in batch for c in b[8]]
        event_id = torch.LongTensor([b[9] for b in batch])
        roles_list = [b[10] for b in batch]
        triggers_head = pad_sequence(triggers_head, batch_first=True, padding_value=0).long()
        triggers_tail = pad_sequence(triggers_tail, batch_first=True, padding_value=0).long()
        batch_size = triggers_head.size(0)
        assert len(arguments_head) % batch_size == 0
        event_role_len = len(arguments_head) // batch_size
        arguments_head = pad_sequence(arguments_head, batch_first=True, padding_value=0).long().view(batch_size, event_role_len, -1)
        arguments_tail = pad_sequence(arguments_tail, batch_first=True, padding_value=0).long().view(batch_size, event_role_len, -1)
        return text, tokens, tokens_id[:, :max_length], seg[:, :max_length], triggers_head[:, :max_length], \
                triggers_tail[:, :max_length], trigger_head, trigger_tail, arguments_head[:, :, :max_length], \
                arguments_tail[:, :, :max_length], event_id, roles_list
    else:
        print("Undefined model type!")
        exit()

if __name__ == "__main__":
    """ # Test for scheme dict
    scheme_dict = SchemaDict()
    print(scheme_dict.event_dict)
    print(scheme_dict.event_list)
    print(scheme_dict.role_dict)
    print(scheme_dict.role_list)
    print(scheme_dict.get_event_id("ef"))
    print(scheme_dict.get_event_type("4"))
    print(scheme_dict.get_role_id(5, "时间"))
    print(scheme_dict.get_role_type(5, 5)) """

    # Test for event dataset
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.K = K
    args.vocab_path = vocabulary_path
    args.spm_model_path = None

    """ event_dataset = EventDataset(args, TRAIN, 0, True)
    event_data_loader = DataLoader(event_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in event_data_loader:
        print(batch[0][0])
        print(batch[1][0])
        print(batch[2][0])
        for i in range(65):
            print(batch[3][0][i])
        print(batch[4][0])
        break """

    """ args.model_type = "hierarchical"
    event_dataset = EventDataset(args, TRAIN, 0, True)
    event_data_loader = DataLoader(event_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: collate_fn(x, model_type=args.model_type))
    for batch in event_data_loader:
        for i in range(11):
            print(batch[i][1])
        break """
    
    args.model_type = "cascade"
    event_dataset = EventDataset(args, TRAIN, 0, True)
    event_data_loader = DataLoader(event_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: collate_fn(x, model_type=args.model_type))
    for batch in event_data_loader:
        for i in range(12):
            print(batch[i][1])
            if i == 8 or i == 9:
                print(np.where(batch[i][1] > 0.5))
        break