import sys

from reformer.reformer_pytorch import Reformer, FixedPositionEmbedding

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
from utils import read_source_text

# instantiate model

def default(val, default_val):
    return default_val if val is None else val

def identity(x):
    return x

class ReformerClassifier(nn.Module):
    def __init__(self, vocab, dim, depth, max_seq_len, num_heads = 8, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False, full_attn_thres = 0, num_mem_kv = 0, return_embeddings = False, fixed_position_emb = False):
        """

        :param vocab: the discharge note and ICD vocabulary
        :param dim: the size of the embedding dimension of each vocabulary word
        :param depth: the depth/number of transformer blocks
        :param max_seq_len: the max length of each input sequence
        :param num_heads: the number of heads for multi-headed attention
        :param bucket_size:
        :param n_hashes:
        :param ff_chunks:
        :param attn_chunks:
        :param causal:
        :param weight_tie:
        :param lsh_dropout:
        :param layer_dropout:
        :param random_rotations_per_head:
        :param twin_attention:
        :param use_scale_norm:
        :param use_full_attn:
        :param full_attn_thres:
        :param num_mem_kv:
        :param return_embeddings:
        :param fixed_position_emb:
        """
        super().__init__()
        emb_dim = dim
        self.vocab = vocab
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.ff_chunks = ff_chunks
        self.attn_chunks = attn_chunks
        self.causal = causal
        self.weight_tie = weight_tie
        self.lsh_dropout = lsh_dropout
        self.layer_dropout = layer_dropout
        self.random_rotations_per_head = random_rotations_per_head
        self.twin_attention = twin_attention
        self.use_scale_norm = use_scale_norm
        self.use_full_attn = use_full_attn
        self.full_attn_thres = full_attn_thres
        self.num_mem_kv = num_mem_kv
        self.return_embeddings = return_embeddings
        self.fixed_position_emb = fixed_position_emb

        num_tokens = len(vocab.discharge) # Number of tokens in the discharge note vocabulary
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.num_output_classes = len(self.vocab.icd)
        self.pos_emb = FixedPositionEmbedding(emb_dim) if fixed_position_emb else nn.Embedding(max_seq_len, emb_dim)
        self.to_model_dim = identity if emb_dim == dim else nn.Linear(emb_dim, dim)

        self.reformer = Reformer(dim, depth, max_seq_len, heads = num_heads, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, num_mem_kv = num_mem_kv)
        self.to_logits = identity if return_embeddings else nn.Linear(dim, num_tokens)
        self.pre_classifier = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, self.num_output_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, source_lengths, **kwargs):
        mask = (x == self.vocab.discharge.pad_token).to(x.device)
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x)
        # x = x + self.pos_emb(t).type(x.type())

        # x = self.to_model_dim(x)
        hidden_state = self.reformer(x, input_mask=mask)  # (bs, seq length, dim)
        pooled_output = hidden_state[:, 0, :]  # (bs, dim) - we take the first character (the CLS token)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = nn.Tanh()(pooled_output)  # (bs, dim)
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
        model = ReformerClassifier(vocab=params['vocab'], **args)
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
            bucket_size=self.bucket_size,
            n_hashes=self.n_hashes,
            ff_chunks=self.ff_chunks,
            attn_chunks=self.attn_chunks,
            causal=self.causal,
            weight_tie=self.weight_tie,
            lsh_dropout=self.lsh_dropout,
            layer_dropout=self.layer_dropout,
            random_rotations_per_head=self.random_rotations_per_head,
            twin_attention=self.twin_attention,
            use_scale_norm=self.use_scale_norm,
            use_full_attn=self.use_full_attn,
            full_attn_thres=self.full_attn_thres,
            num_mem_kv=self.num_mem_kv,
            return_embeddings=self.return_embeddings,
            fixed_position_emb=self.fixed_position_emb
        )

        params = {
            'args': args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
