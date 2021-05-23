#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.05.18

import os

""" Check if the dir of file_path exists. If not, create it. """
def check_file_path(file_path):
    dir_name = os.path.dirname(file_path)
    if os.path.exists(dir_name): return
    os.makedirs(dir_name)