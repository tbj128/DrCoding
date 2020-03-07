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
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
# import nltk
# nltk.download('punkt')


def create_embedding_from_glove(glove_path, vocab, device):
    print("Creating embedding from GLOVE...")
    glove = {}
    embed_size = 0
    with open(glove_path, 'r') as f:
        i = 0
        for line in f:
            row = line.split(" ")
            word = row[0]
            embedding = [float(x) for x in row[1:]]
            if word in vocab.discharge.word2id:
                glove[word] = embedding
            embed_size = len(embedding)
            i += 1
            # print("   > Completed {}".format(i), end='\r', flush=True)
            # if i > 1000:
            #     break

    weights_matrix = np.zeros((len(vocab.discharge), embed_size))
    words_found = 0
    for word, index in vocab.discharge.word2id.items():
        try:
            weights_matrix[index] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[index] = np.random.normal(scale=0.6, size=(embed_size,))
    print("Number of words found in glove {}/{}".format(words_found, len(vocab.discharge)))

    weights_matrix = torch.tensor(weights_matrix, device=device)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


def read_source_text(file_path, target_length=1000, pad_token='<pad>', use_cls=False):
    """
    Read the input discharge summaries.
    Take the first target_length number of words or pad with pad_token if the summary is less than the target_length
    @param file_path (str): path to file containing discharge summaries
    @param target_length (str): the length that the sentence should be
    @param pad_token (str): the padding token
    @param use_cls (bool): true if we should add a classification token
    """
    data = []
    icds = []
    source_lengths = []

    data_df = pd.read_csv(file_path, header=None)
    for (i, row) in enumerate(data_df.values):
        hadmid = row[0]
        line = row[1]
        row_icds = row[2:]

        icds.append(row_icds)

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

    assert len(data) == len(source_lengths)
    assert len(source_lengths) == len(icds)

    return data, source_lengths, icds


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
        #
        # examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_text = [e[0] for e in examples]
        source_lengths = [e[1] for e in examples]
        icd_codes = [e[2] for e in examples]

        yield src_text, source_lengths, icd_codes
