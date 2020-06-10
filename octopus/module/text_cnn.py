#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_class: int):
        super(TextCNN, self).__init__()
        self.pe = PositionalEncoding(embedding_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.enc = nn.Sequential(
            conv_block(embedding_dim, hidden_dim, 4),
            conv_block(hidden_dim, hidden_dim, 3),
            conv_block(hidden_dim, hidden_dim, 3),
            conv_block(hidden_dim, n_class, 2)
        )

    def forward(self, inputs):
        embeds = self.embeddings(inputs)  # (N * S * E)
        embeds += self.pe(embeds)  # (N * S * E)
        embeds = embeds.transpose(2, 1)
        out = self.enc(embeds).transpose(1, 2) # (N * S * E)
        nll_prob = F.log_softmax(out, dim=-1)
        # return: [N * n_class]
        return nll_prob


def conv_block(in_channels, out_channels, k_size):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )
