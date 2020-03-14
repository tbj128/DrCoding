#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Input manipulation for the MIMIC-III ICD-9 prediction task

Utilities to read and iterate over the input files

"""
import csv
import math
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
# import nltk
# nltk.download('punkt')
from bert.bert_utils import convert_examples_to_features, InputExample


def create_embedding_from_glove(glove_path, vocab, device):
    """
    Creates an embedding model based on a pre-trained GLOVE embedding file
    :param glove_path: the path to the GLOVE model
    :param vocab: the vocab
    :param device: the Torch device
    :return: emb_layer, num_embeddings, embedding_dim
    """
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
    emb_layer.weight.requires_grad = True

    return emb_layer, num_embeddings, embedding_dim


def read_source_text_for_bert_with_metadata(file_path, tokenizer, metadata_file_path=None, target_length=1000):
    """
    Read the input discharge summaries, but massage the data into a model that can be more easily
    passed to the HuggingFace transformer

    Take the last target_length number of words

    @param file_path (str): path to file containing discharge summaries
    @param tokenizer (str): the BERT tokenizer
    @param metadata_file_path (str): path to file containing descriptions of the ICD codes for each discharge summaries
    @param target_length (str): the length that the sentence should be
    """

    hadmid_to_metadata = {}
    if metadata_file_path is not None and metadata_file_path != "NONE":
        with open(metadata_file_path, 'r') as f:
            metadata_tsv = csv.reader(f, delimiter='\t')
            i = 0
            for row in metadata_tsv:
                hadmid = int(row[0])
                descriptions = " ".join(row[1:])
                hadmid_to_metadata[hadmid] = descriptions
                i += 1

    examples = []
    icds = []
    icd_descriptions = []
    source_lengths = []
    data_df = pd.read_csv(file_path, header=None)
    for (i, row) in enumerate(data_df.values):
        hadmid = row[0]

        line = row[1].split(" ")
        line = " ".join(line[:target_length])

        row_icds = row[2:]

        if metadata_file_path is not None and metadata_file_path != "NONE":
            row_icd_descriptions = hadmid_to_metadata[hadmid]
        else:
            row_icd_descriptions = ""
        # row_icd_descriptions = row_icd_descriptions.split(" ")
        # if len(row_icd_descriptions) > target_length:
        #     row_icd_descriptions = row_icd_descriptions[:target_length]
        #
        # orig_description = list(row_icd_descriptions)
        # while len(row_icd_descriptions) < target_length:
        #     # Why don't we replicate the metadata over and over to emphasize?
        #     row_icd_descriptions.extend(orig_description)
        # row_icd_descriptions = " ".join(row_icd_descriptions[:target_length])

        if i == 0:
            print("Input example: {} {} {} {}".format(hadmid, line[0:100], row_icds, row_icd_descriptions))

        icds.append(row_icds)
        icd_descriptions.append(row_icd_descriptions)

        sent = line.split(" ")
        length = len(sent)
        source_lengths.append(length)

        examples.append(InputExample(guid=hadmid, text=line, metadata_text=row_icd_descriptions))

    # The max metadata length is set to be the same as the target length. Although the metadata length is
    # usually much smaller than the target length, the target needs to be passed during test time.
    features = convert_examples_to_features(examples, target_length, tokenizer, target_length)

    return features, source_lengths, icds, icd_descriptions


def read_icd_descs_for_testing(f_icdmap, top_icds, device, metadata_len, tokenizer=None):
    icd_to_icd_pos = {}
    icd_pos_to_icd_desc = {}
    final_output = []

    with open(top_icds, 'r') as f:
        metadata_tsv = csv.reader(f, delimiter='\t')
        i = 0
        for row in metadata_tsv:
            icd_to_icd_pos[row[0]] = i
            i += 1

    with open(f_icdmap, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None) # Skip headers
        for line in reader:
            condensed_icd = line[1][:3]
            if condensed_icd in icd_to_icd_pos:
                icd_pos_to_icd_desc[icd_to_icd_pos[condensed_icd]] = line[3]
    for k in range(50):
        tokens_metadata_text = tokenizer.tokenize(icd_pos_to_icd_desc[k])
        metadata_ids = tokenizer.convert_tokens_to_ids(tokens_metadata_text)
        metadata_ids = metadata_ids[:metadata_len]
        metadata_ids += [0] * (metadata_len - len(metadata_ids))
        final_output.append(metadata_ids)
    return torch.tensor(final_output, device=device)


def read_source_text(file_path, metadata_file_path, target_length=1000, pad_token='<pad>', use_cls=False, use_tail=True, notes_delimiter=","):
    """
    Read the input discharge summaries.
    Take the last target_length number of words or pad with pad_token if the summary is less than the target_length

    @param file_path (str): path to file containing discharge summaries
    @param metadata_file_path (str): path to file containing descriptions of the ICD codes for each discharge summaries
    @param target_length (str): the length that the sentence should be
    @param pad_token (str): the padding token
    @param use_cls (bool): true if we should add a classification token
    @param use_tail (bool): true if we should use the words from the end of the text rather than the beginning
    """
    data = []
    icds = []
    icd_descriptions = []
    source_lengths = []

    hadmid_to_metadata = {}
    if metadata_file_path is not None and metadata_file_path != "NONE":
        with open(metadata_file_path, 'r') as f:
            metadata_tsv = csv.reader(f, delimiter='\t')
            i = 0
            for row in metadata_tsv:
                hadmid = int(row[0])
                descriptions = " ".join(row[1:])
                hadmid_to_metadata[hadmid] = descriptions
                i += 1

    data_df = pd.read_csv(file_path, header=None, delimiter=notes_delimiter)
    for (i, row) in enumerate(data_df.values):
        hadmid = row[0]

        line = row[1].split(" ")
        if use_tail:
            line = " ".join(line[-target_length:]) # prefer to take the last words
        else:
            line = " ".join(line[:target_length]) # prefer to take the first words


        row_icds = row[2:]
        if metadata_file_path is not None and metadata_file_path != "NONE":
            row_icd_descriptions = hadmid_to_metadata[hadmid]
        else:
            row_icd_descriptions = ""
        # row_icd_descriptions = row_icd_descriptions.split(" ")
        # if len(row_icd_descriptions) > target_length:
        #     row_icd_descriptions = row_icd_descriptions[:target_length]
        # orig_description = list(row_icd_descriptions)
        # while len(row_icd_descriptions) < target_length:
        #     # Why don't we replicate the metadata over and over to emphasize?
        #     row_icd_descriptions.extend(orig_description)
        # row_icd_descriptions = row_icd_descriptions[:target_length]

        icds.append(row_icds)
        icd_descriptions.append(row_icd_descriptions)

        sent = line.split(" ")
        if use_cls:
            sent = ['<cls>'] + sent
        length = len(sent)
        if len(sent) > target_length:
            sent = sent[:target_length]
            length = target_length
        while len(sent) < target_length:
            sent.append(pad_token)
        data.append(sent)
        source_lengths.append(length)

    assert len(data) == len(source_lengths)
    assert len(source_lengths) == len(icds)

    return data, source_lengths, icds, icd_descriptions


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
        icd_code_descs = [e[3] for e in examples]

        yield src_text, source_lengths, icd_codes, icd_code_descs
