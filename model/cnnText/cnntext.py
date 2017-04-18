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


    input_channel = args['input_channels']
    output_channel = args['output_channels']
    target_class = args['target_class']
    words_num = args['words_num']
    words_dim = args['words_dim']
    embeds_num = args['embeds_num']
    embeds_dim = args['embeds_dim']
    Ks = args['kernel_sizes']
    self.embed = nn.Embedding(words_num, words_dim)
    self.static_embed = nn.Embedding(embeds_num, embeds_dim)
    self.static_embed.weight.data.copy_(torch.from_numpy(args['embeds']))
    self.static_embed.weight.requires_grad = False
    self.convs1 = [nn.Conv2d(input_channel, output_channel, (K, words_dim)) for K in Ks]

    self.dropout = nn.Dropout(args['dropout'])
    self.fc1 = nn.Linear(len(Ks) * output_channel, target_class)

  def conv_and_pool(self, x, conv):
    x = F.relu(conv(x)).squeeze(3) # (batch_size, output_channel, feature_map_dim)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x

  def forward(self, x):
    words = x[:,:,0]
    word_input = self.embed(words) # (batch, sent_len, embed_dim)
    static_input = self.static_embed(words)
    x = torch.stack([word_input, static_input], dim=1)# (batch, channel_input, sent_len, embed_dim)
    #x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
    # (batch, channel_output, ~=(sent_len)) * len(Ks)
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
    # (batch, channel_output) * len(Ks)
    x = torch.cat(x, 1) # (batch, channel_output * len(Ks))
    x = self.dropout(x)
    logit = self.fc1(x) # (batch, target_size)
    return logit

