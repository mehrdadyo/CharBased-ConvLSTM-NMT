#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size: int, word_embed_size : int, window_size=5):
        super().__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.window_size = window_size

        self.conv1D = nn.Conv1d(self.char_embed_size, self.word_embed_size, kernel_size=window_size, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv1D.forward(x)
        x_con_relu = torch.relu(x_conv)
        # 1D max pooling
        x_conv_out = torch.max(x_con_relu, dim=2)[0]

        return x_conv_out




    ### END YOUR CODE

