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
from utils import create_embedding_from_glove


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
    def __init__(self, vocab, embed_size, hidden_size, depth, max_seq_len, device, num_heads = 8, layer_dropout = 0., fixed_position_emb = False, glove_path=None):
        """
        """
        super().__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.layer_dropout = layer_dropout
        self.fixed_position_emb = fixed_position_emb
        self.device = device
        self.glove_path = glove_path
        self.num_output_classes = len(self.vocab.icd)

        num_tokens = len(vocab.discharge) # Number of tokens in the discharge note vocabulary

        if glove_path is not None and glove_path != "NONE":
            emb_layer, num_embeddings, embedding_dim = create_embedding_from_glove(glove_path, self.vocab, device)
            self.embeddings = emb_layer
        else:
            self.embeddings = nn.Embedding(len(self.vocab.discharge), embed_size, padding_idx=self.vocab.discharge.pad_token)

        encoder_layer = nn.TransformerEncoderLayer(d_model=max_seq_len, nhead=num_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pre_classifier = nn.Linear(max_seq_len, max_seq_len)
        self.classifier = nn.Linear(max_seq_len, self.num_output_classes)
        self.dropout = nn.Dropout(self.layer_dropout)


    def forward(self, x, source_lengths, **kwargs):
        # x batch size must be in the middle
        # mask = self.generate_sent_masks(x, source_lengths)

        # t = torch.arange(x.shape[1], device=x.device)
        x = self.embeddings(x)
        x = x.permute(1, 0, 2).contiguous()

        # x = x + self.pos_emb(t).type(x.type())

        # x = self.to_model_dim(x) # (bs, seq length, dim)
        # hidden_state = self.transformer_encoder(x, mask=mask, **kwargs)  # (bs, seq length, dim)
        hidden_state = self.transformer_encoder(x, **kwargs)  # (seq length, bs, dim)
        # hidden_state = x

        pooled_output = hidden_state[0, :, :]  # (bs, dim) - we take the first character (the CLS token)
        # pooled_output = torch.mean(hidden_state, dim=0)


        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        # pooled_output = nn.Tanh()(pooled_output)  # (bs, dim)
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
            emb_dim=self.emb_dim,
            depth=self.depth,
            hidden_size=self.hidden_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            layer_dropout=self.layer_dropout,
            fixed_position_emb=self.fixed_position_emb,
            device=self.device,
            glove_path=self.glove_path

        )

        params = {
            'args': args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def generate_sent_masks(self, x, source_lengths) -> torch.Tensor:
        enc_masks = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:, src_len:] = float('-inf') # Mask padding
        # enc_masks = enc_masks.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return enc_masks

