#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

import json

from event_extraction_codes.config import *

""" Check if the dir of file_path exists. If not, create it. """
def check_file_path(file_path):
    dir_name = os.path.dirname(file_path)
    if os.path.exists(dir_name): return
    os.makedirs(dir_name)

def make_event_corpus():
    input_path = origin_train_path
    output_path = corpus_path
    check_file_path(output_path)
    fw = open(output_path, "w", encoding="utf-8")
    with open(input_path, "r", encoding="utf-8") as fr:
        for line_id, line in enumerate(fr):
            line = json.loads(line)
            text = line["text"]
            text_1 = text[0:len(text)//2] + "\n"
            text_2 = text[len(text)//2:] + "\n"
            fw.write(text_1 + text_2 + "\n")

if __name__ == "__main__":
    make_event_corpus()