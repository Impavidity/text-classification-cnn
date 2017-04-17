#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import torch.nn.functional as F

from configurable import Configurable

class CNNText(nn.Module):
  """
  Model class for the computational graph
  """
  def __init__(self, args):
    super(CNNText, self).__init__()
    self.args = args

    input_channel = args.input_channels
    output_channel = args.output_channels
    target_class = args.target_class
    embed_num = args.embed_num
    embed_dim = args.embed_dim
    Ks = args.kernel_sizes
    self.embed = nn.Embedding(embed_num, embed_dim)
    self.convs1 = [nn.Conv2d(input_channel, output_channel, (K, embed_dim)) for K in Ks]

    self.dropout = nn.Dropout(args.dropout)
    self.fc1 = nn.Linear(len(Ks) * output_channel, target_class)

  def conv_and_pool(self, x, conv):
    x = F.relu(conv(x)).squeeze(3) # (batch_size, output_channel, feature_map_dim)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x

  def forward(self, x):
    word_input = self.embed(x) # (batch, sent_len, embed_dim)
    x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
    # (batch, channel_output, ~=(sent_len)) * len(Ks)
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
    # (batch, channel_output) * len(Ks)
    x = torch.cat(x, 1) # (batch, channel_output * len(Ks))
    x = self.dropout(x)
    logit = self.fc1(x) # (batch, target_size)
    return logit


