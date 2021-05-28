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

        self.event_dict, self.event_list = {}, []
        self.__init_event__()
        self.__init_role__()
        
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

        if not os.path.exists(self.preprocessed_data_path) or update is True:
            self.__preprocess_data_baseline__()
        
        self.__get_data__()
    
    def __preprocess_data_baseline__(self):
        """ Generate the preprocessed data. """
        print("Preprocessing dataset...")
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

                        # Calculating the overlap number. Train: 1492.
                        """ if np.any(np.array(tags)[:, argument_start_index:argument_end_index]):
                            #print(np.array(tags)[:, argument_start_index:argument_end_index])
                            is_overlap = True """
                        
                        self.__sequence_tag__(argument_start_index, argument_end_index, str(role_label), tags[event_type_id])
                preprocessed_sample = json.dumps({"text": text, "tokens": tokens, "tokens_id": tokens_id, "tags": tags}, ensure_ascii=False)
                # if is_overlap is True: self.overlap_num += 1
                # if idx == 0: print(preprocessed_sample)
                fw.write(preprocessed_sample + "\n")

        #print("Max length: {}, overlap_num: {}.".format(self.max_length, self.overlap_num))
        fw.close()

    def __get_data__(self):
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
    
    def __getitem__(self, index):
        return self.text[index], self.tokens[index], self.tokens_id[index], self.tags[index]

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

def collate_fn(batch, max_len=512):
    """ 
        Collate_fn function. Arrange the batch in reverse order of length.

        Args:

            batch:              The batch data.
            max_len:            The max length.
        
        Returns:
            ([text], [tokens], [tokens_id], [tags], [seg])
    """
    data_length = [len(b[1]) for b in batch]
    max_length = min(max(data_length), max_len)
    text, tokens = [b[0] for b in batch], [b[1] for b in batch]
    tokens_id = [torch.FloatTensor(b[2]) for b in batch]
    tags = [torch.FloatTensor(b[3]).T for b in batch]
    #print(tags[0].shape)
    seg = [torch.FloatTensor([1] * _) for _ in data_length]
    tokens_id = pad_sequence(tokens_id, batch_first=True, padding_value=PAD_ID).long()
    tags = torch.transpose(pad_sequence(tags, batch_first=True, padding_value=LABEL_P).long(), 1, 2)
    seg = pad_sequence(seg, batch_first=True, padding_value=0).long()
    #print(tags.shape)
    return text, tokens, tokens_id[:, :max_length], tags[:, :, :max_length], seg[:, :max_length]

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

    event_dataset = EventDataset(args, TEST, 0, True)
    event_data_loader = DataLoader(event_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in event_data_loader:
        print(batch[0][0])
        print(batch[1][0])
        print(batch[2][0])
        for i in range(65):
            print(batch[3][0][i])
        print(batch[4][0])
        break