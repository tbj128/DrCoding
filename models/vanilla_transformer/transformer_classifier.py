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
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_source_lengths, has_mask=False):
        mask = (src == self.vocab.discharge.pad_token).to(src.device)
        src = src.permute(1, 0)

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask) # (seq_len, bs, ninp)
        output = output[0, :, :].squeeze()  # (bs, ninp) - we take the first character (the CLS token)

        # pooled_output = torch.mean(hidden_state, dim=0)
        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        # pooled_output = nn.Tanh()(pooled_output)  # (bs, dim)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # logits = self.classifier(pooled_output)  # (bs, dim)

        return self.classifier(output)
        # return F.log_softmax(output, dim=-1)
#
#
#
#
#
# class FixedPositionEmbedding(nn.Module):
#     ## TODO: Merge with reformer
#     def __init__(self, dim):
#         super().__init__()
#         inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2) / dim))
#         self.register_buffer('inv_freq', inv_freq)
#
#     def forward(self, positions):
#         sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
#         emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
#         return emb[None, :, :]
#
# class TransformerClassifier(nn.Module):
#     def __init__(self, vocab, embed_size, hidden_size, depth, max_seq_len, device, num_heads = 8, layer_dropout = 0., fixed_position_emb = False, glove_path=None):
#         """
#         """
#         super().__init__()
#         self.vocab = vocab
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.depth = depth
#         self.max_seq_len = max_seq_len
#         self.num_heads = num_heads
#         self.layer_dropout = layer_dropout
#         self.fixed_position_emb = fixed_position_emb
#         self.device = device
#         self.glove_path = glove_path
#         self.num_output_classes = len(self.vocab.icd)
#
#         num_tokens = len(vocab.discharge) # Number of tokens in the discharge note vocabulary
#
#         if glove_path is not None and glove_path != "NONE":
#             emb_layer, num_embeddings, embedding_dim = create_embedding_from_glove(glove_path, self.vocab, device)
#             self.embeddings = emb_layer
#         else:
#             self.embeddings = nn.Embedding(len(self.vocab.discharge), embed_size, padding_idx=self.vocab.discharge.pad_token)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=max_seq_len, nhead=num_heads, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#
#         self.pre_classifier = nn.Linear(max_seq_len, max_seq_len)
#         self.classifier = nn.Linear(max_seq_len, self.num_output_classes)
#         self.dropout = nn.Dropout(self.layer_dropout)
#
#
#     def forward(self, x, source_lengths, **kwargs):
#         # x batch size must be in the middle
#         # mask = self.generate_sent_masks(x, source_lengths)
#
#         # t = torch.arange(x.shape[1], device=x.device)
#         x = self.embeddings(x)
#         x = x.permute(1, 0, 2).contiguous()
#
#         # x = x + self.pos_emb(t).type(x.type())
#
#         # x = self.to_model_dim(x) # (bs, seq length, dim)
#         # hidden_state = self.transformer_encoder(x, mask=mask, **kwargs)  # (bs, seq length, dim)
#         hidden_state = self.transformer_encoder(x, **kwargs)  # (seq length, bs, dim)
#         # hidden_state = x
#
#         pooled_output = hidden_state[0, :, :]  # (bs, dim) - we take the first character (the CLS token)
#         # pooled_output = torch.mean(hidden_state, dim=0)
#
#
#         pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
#         pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
#         # pooled_output = nn.Tanh()(pooled_output)  # (bs, dim)
#         pooled_output = self.dropout(pooled_output)  # (bs, dim)
#         logits = self.classifier(pooled_output)  # (bs, dim)
#
#         return logits

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
