#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | LSTM baseline model for the MIMIC-III ICD-9 prediction task
"""
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f


class DischargeLSTM(nn.Module):
    def __init__(self, vocab, hidden_size, dropout_rate, embed_size, pretrained_embeddings=None):
        """

        Bidrectional LSTM Multi-Label Classifier with Glove embeddings
        :param vocab:
        :param hidden_size:
        :param dropout_rate:
        :param num_output_classes:
        :param embed_size:
        :param pretrained_embeddings:
        """

        super(DischargeLSTM, self).__init__()

        self.pretrained_embeddings = pretrained_embeddings
        self.vocab = vocab
        self.num_output_classes = len(self.vocab.icd)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size

        if pretrained_embeddings is not None:
            weights_matrix = np.zeros((len(self.vocab.discharge), pretrained_embeddings.embed_size))
            self.embed_size = pretrained_embeddings.embed_size
            for w in self.vocab.discharge:
                weights_matrix[i] = glove[w]
            self.embeddings = nn.Embedding(len(self.vocab.discharge), embed_size, padding_idx=self.vocab.discharge.pad_token)
            self.embeddings.load_state_dict({'weight': weights_matrix})
            self.embeddings.weight.requires_grad = False
        else:
            self.embed_size = embed_size
            self.embeddings = nn.Embedding(len(self.vocab.discharge), embed_size, padding_idx=self.vocab.discharge.pad_token)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True, bias=True)
        self.linear = nn.Linear(embed_size * 2, self.num_output_classes, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, discharge_padded, source_lengths):
        """
        Forward pass of the LSTM with a linear layer
        :param discharge_padded: minibatch of discharge notes (batch_size, max_discharge_size)
        :return:
        """

        embeddings = self.embeddings(discharge_padded)
        discharge_padded_packed = nn.utils.rnn.pack_padded_sequence(embeddings, source_lengths, batch_first=True)
        output_state, hidden_and_cell = self.lstm(discharge_padded_packed)
        final_hidden, final_cell = hidden_and_cell # (2, batch_size, embed_size)
        final_hidden = final_hidden.permute(1, 0, 2).contiguous() # (batch_size, 2, embed_size)
        final_hidden = final_hidden.view(final_hidden.shape[0], -1) # (batch_size, 2 * embed_size)
        lstm_out = self.linear(final_hidden)
        lstm_out = self.dropout(lstm_out) # batch_size x num_output_classes
        return lstm_out

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
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
            'args': dict(hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, embed_size=self.embed_size),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
