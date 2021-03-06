#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding
LSTM baseline model for the MIMIC-III ICD-9 prediction task
"""
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from utils import create_embedding_from_glove


class DischargeLSTM(nn.Module):
    def __init__(self, vocab, hidden_size, dropout_rate, embed_size, device, glove_path=None):
        """
        Bidrectional LSTM Multi-Label Classifier with Glove embeddings
        :param vocab: the vocab object
        :param hidden_size: the hidden size of the LSTM
        :param dropout_rate: the dropout to apply after the LSTM
        :param num_output_classes: the number of ICDs to predict
        :param embed_size: the size of the word embeddings
        :param glove_path: the path to the GLOVE file
        """

        super(DischargeLSTM, self).__init__()

        self.glove_path = glove_path
        self.vocab = vocab
        self.num_output_classes = len(self.vocab.icd)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.device = device

        if glove_path is not None and glove_path != "NONE":
            emb_layer, num_embeddings, embedding_dim = create_embedding_from_glove(glove_path, self.vocab, device)
            self.embeddings = emb_layer
            self.embed_size = embedding_dim
        else:
            self.embed_size = embed_size
            self.embeddings = nn.Embedding(len(self.vocab.discharge), embed_size, padding_idx=self.vocab.discharge.pad_token)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, bidirectional=True, bias=True)
        self.linear = nn.Linear(embed_size * 2, self.num_output_classes, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, discharge_padded, source_lengths):
        """
        Forward pass of the LSTM with a linear layer
        :param discharge_padded: the padded discharge summaries (bs, seq length)
        :param source_lengths: the actual, original discharge summaries length (bs)
        :return: logits after applying the model (bs, num_output_classes)
        """

        embeddings = self.embeddings(discharge_padded)
        discharge_padded_packed = nn.utils.rnn.pack_padded_sequence(embeddings, source_lengths, batch_first=True, enforce_sorted=False)
        output_state, hidden_and_cell = self.lstm(discharge_padded_packed)
        final_hidden, final_cell = hidden_and_cell # (2, batch_size, embed_size)
        final_hidden = final_hidden.permute(1, 0, 2).contiguous() # (batch_size, 2, embed_size)
        final_hidden = final_hidden.view(final_hidden.shape[0], -1) # (batch_size, 2 * embed_size)
        final_hidden = self.dropout(final_hidden)
        lstm_out = self.linear(final_hidden) # batch_size x num_output_classes
        return lstm_out

    @staticmethod
    def load(model_path: str):
        """
        Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = DischargeLSTM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """
        Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, embed_size=self.embed_size, device=self.device, glove_path=self.glove_path),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
