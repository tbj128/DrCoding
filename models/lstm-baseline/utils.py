#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')


def read_source_text(file_path, target_length, pad_token):
    """
    Read the input discharge summaries.
    Take the first target_length number of words or pad with pad_token if the summary is less than the target_length
    @param file_path (str): path to file containing discharge summaries
    @param target_length (str): the length that the sentence should be
    @param pad_token (str): the padding token
    """
    data = []
    with open(file_path) as f:
        for line in f:
            sent = nltk.word_tokenize(line)
            if len(sent) > target_length:
                sent = sent[:target_length]
            while len(sent) < target_length:
                sent.append(pad_token)
            data.append(sent)
    return data


def read_icd_codes(file_path):
    """
    Read the ICD codes into a one-hot like vector
    @param file_path (str): path to file containing ICD codes
    """
    icd_to_pos = {}
    pos_to_icd = {}
    pos = 0
    with open(file_path) as f:
        for line in f:
            for icd in line.split(","):
                if icd not in icd_to_pos:
                    icd_to_pos[icd] = pos
                    pos_to_icd[pos] = icd
                    pos += 1
    data = []
    with open(file_path) as f:
        for line in f:
            out_arr = [0] * len(icd_to_pos)
            for icd in line.split(","):
                out_arr[icd_to_pos[icd]] = 1
            data.append(out_arr)
    return data, pos_to_icd


def batch_iter(data, batch_size, shuffle=False):
    """
    Yield batches of source text and target ICD codes reverse-sorted by length (largest to smallest).
    @param data (list of (src_text, icd_codes)): list of tuples containing source text and list of ICD codes
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_text = [e[0] for e in examples]
        icd_codes = [e[1] for e in examples]

        yield src_text, icd_codes
