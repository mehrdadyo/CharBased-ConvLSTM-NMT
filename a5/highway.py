#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embed_size: int):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)

    def forward(self, x_conv_out : torch.Tensor) -> torch.Tensor:
        x_proj = torch.relu(self.proj.forward(x_conv_out))
        x_gate = torch.sigmoid(self.gate.forward(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul(1 - x_gate, x_proj)

        return x_highway






    ### END YOUR CODE

