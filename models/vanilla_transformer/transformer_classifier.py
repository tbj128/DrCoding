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
    """
    Transformer-based classifier model
    """

    def __init__(self, vocab, ninp, nhead, nhid, nlayers, dropout=0.1):
        """
        Constructor.
        :param vocab: the vocab object
        :param ninp: size of the word embedding
        :param nhead: number of Transformer heads
        :param nhid: the hidden size of each Transformer layer
        :param nlayers: number of Transformer layers
        :param dropout: the dropout to apply in each Transformer layer
        """
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

    def forward(self, src, batch_source_lengths):
        """
        The forward pass of the Transformer
        :param src: the padded input text (bs, seq len)
        :param batch_source_lengths: the lengths of the original input text (bs, 1)
        :return: logits of the Transformer (bs, num_output_classes)
        """

        mask = (src == self.vocab.discharge.pad_token)
        src = src.permute(1, 0)

        src = self.encoder(src)

        #
        # Uncomment below if we want to use the positional encoding
        # Empirical evidence suggests position encoding does not
        # perform as well in sequence classification problems
        #
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        hidden_state = self.transformer_encoder(src, src_key_padding_mask=mask) # (seq_len, bs, ninp)

        #
        # Uncomment below if we want to use the CLS token to perform classification
        # Testing shows that the CLS token does not perform as well as if we had
        # took the mean over the hidden states
        #
        # hidden_state = hidden_state[0, :, :].squeeze()  # (bs, ninp) - we take the first character (the CLS token)

        # Take the masked mean over the output hidden states. Note that the mean
        # is only calculated for positions which did not correspond to a padding token
        hidden_state = hidden_state * (~mask).type(torch.float).transpose(0, 1).unsqueeze(2)
        hidden_state = torch.sum(hidden_state, dim=0) / torch.sum(mask == False, dim=1).type(torch.float).unsqueeze(1)

        pooled_output = self.pre_classifier(hidden_state)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_output_classes)

        return logits

    @staticmethod
    def load(model_path: str):
        """
        Load the model from a file.
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
