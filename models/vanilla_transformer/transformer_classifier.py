import sys

import random
import tqdm
import gzip
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformer_common.positional_encoding import PositionalEncoding
from vocab import Vocab
from utils import create_embedding_from_glove


class TransformerClassifier(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, vocab, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.vocab = vocab
        self.ntoken = len(vocab.discharge)
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_output_classes = len(vocab.icd)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(self.ntoken, ninp)
        self.classifier = nn.Linear(ninp, self.num_output_classes)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_source_lengths, has_mask=False):
        mask = (src == self.vocab.discharge.pad_token)
        src = src.permute(1, 0)

        src = self.encoder(src)
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        hidden_state = self.transformer_encoder(src, src_key_padding_mask=mask) # (seq_len, bs, ninp)
        # output = output[0, :, :].squeeze()  # (bs, ninp) - we take the first character (the CLS token)

        hidden_state = hidden_state * (~mask).type(torch.float).transpose(0, 1).unsqueeze(2)
        hidden_state = torch.sum(hidden_state, dim=0) / torch.sum(mask == False, dim=1).unsqueeze(1)

        pooled_output = self.pre_classifier(hidden_state)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TransformerClassifier(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """
        Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        args = dict(
            ninp=self.ninp,
            nhead=self.nhead,
            nhid=self.nhid,
            nlayers=self.nlayers,
            dropout=self.dropout
        )

        params = {
            'args': args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
