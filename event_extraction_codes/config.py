#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from itertools import chain

""" Dataset """
K = 5                   # k fold.
TRAIN, VALID, TEST = 0, 1, 2
vocabulary_path = os.path.join(uer_dir, "models/google_zh_vocab.txt")
origin_train_path = os.path.join(uer_dir, "datasets/event_extraction_dataset/train.json")
origin_test_path = os.path.join(uer_dir, "datasets/event_extraction_dataset/test.json")
origin_schema_path = os.path.join(uer_dir, "datasets/event_extraction_dataset/event_schema.json")
preprocessed_train_path = os.path.join(uer_dir, "datasets/event_extraction_dataset/train_pre.json")
preprocessed_test_path = os.path.join(uer_dir, "datasets/event_extraction_dataset/test_pre.json")
corpus_path = os.path.join(uer_dir, "corpora/event_corpus.txt")

""" Labels list """
role_label_list = [
    "O", "P"        # Others, Padding
] + list(chain.from_iterable([["S-" + str(_), "B-" + str(_), "I-"+ str(_), "E-" + str(_)] for _ in range(6)]))
LABEL_O, LABEL_P = 0, 1

""" Criterion """
# The weight of O, P and others.
baseline_criterion_weigth = [1.0, 0.0, 50.0]

""" Training settings. """
batch_size = 8
seq_length = 512
report_steps = 250
epochs_num = 100
eval_epoch = 20
seed = 7
#scheduler = "polynomial"
power = 2

""" Files """
pretrained_model_path = os.path.join(uer_dir, "models/bert/event_bert_model.bin")
config_path = os.path.join(uer_dir, "models/bert_base_config.json")