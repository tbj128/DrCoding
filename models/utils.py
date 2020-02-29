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


def read_source_text(file_path, target_length=1000, pad_token='<pad>', use_cls=False):
    """
    Read the input discharge summaries.
    Take the first target_length number of words or pad with pad_token if the summary is less than the target_length
    @param file_path (str): path to file containing discharge summaries
    @param target_length (str): the length that the sentence should be
    @param pad_token (str): the padding token
    """
    data = []
    source_lengths = []
    with open(file_path) as f:
        for line in f:
            # sent = nltk.word_tokenize(line)
            sent = line.split(" ")
            if use_cls:
                sent = ['<cls>'] + sent
            length = len(sent)
            if len(sent) > target_length:
                sent = sent[:target_length]
                length = target_length
            while len(sent) < target_length:
                sent.append(pad_token)
                length += 1
            data.append(sent)
            source_lengths.append(length)
    return data, source_lengths


def read_icd_codes(file_path):
    """
    Read the ICD codes into a one-hot like vector
    @param file_path (str): path to file containing ICD codes
    """
    icds = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            out = []
            for icd in line.split(","):
                out.append(icd.strip())
            icds.append(out)
    return icds


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
        source_lengths = [e[1] for e in examples]
        icd_codes = [e[2] for e in examples]

        yield src_text, source_lengths, icd_codes
