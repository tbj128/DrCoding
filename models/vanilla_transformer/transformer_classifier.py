import sys

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from vocab import Vocab
from utils import read_source_text, read_icd_codes


def identity(x):
    return x

class FixedPositionEmbedding(nn.Module):
    ## TODO: Merge with reformer
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab, dim, depth, max_seq_len, num_heads = 8, layer_dropout = 0., fixed_position_emb = False):
        """
        """
        super().__init__()
        emb_dim = dim
        self.vocab = vocab
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.layer_dropout = layer_dropout
        self.fixed_position_emb = fixed_position_emb

        num_tokens = len(vocab.discharge) # Number of tokens in the discharge note vocabulary
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.num_output_classes = len(self.vocab.icd)
        self.pos_emb = FixedPositionEmbedding(emb_dim) if fixed_position_emb else nn.Embedding(max_seq_len, emb_dim)
        self.to_model_dim = identity if emb_dim == dim else nn.Linear(emb_dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=self.layer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pre_classifier = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, self.num_output_classes)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x, **kwargs):
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x)
        x = x + self.pos_emb(t).type(x.type())

        x = self.to_model_dim(x)
        hidden_state = self.transformer_encoder(x, **kwargs)  # (bs, seq length, dim)
        pooled_output = hidden_state[:, 0, :]  # (bs, dim) - we take the first character (the CLS token)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
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
            dim=self.dim,
            depth=self.depth,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            layer_dropout=self.layer_dropout,
            fixed_position_emb=self.fixed_position_emb
        )

        params = {
            'args': args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
