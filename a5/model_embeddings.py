#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        pad_token_idx = vocab.char2id['∏']
        char_embed_size, window_size, p_dropout = 50, 5, 0.3

        self.char_embedding = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=char_embed_size,
            padding_idx=pad_token_idx
        )

        self.cnn = CNN(char_embed_size=char_embed_size, word_embed_size=word_embed_size, window_size=window_size)
        self.highway = Highway(word_embed_size=word_embed_size)
        self.dropout = nn.Dropout(p=p_dropout)
        self.word_embed_size = word_embed_size
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        x_embed = self.char_embedding.forward(input)
        max_sent_length, batch_size, max_word_length, char_embed_size = x_embed.size()

        x_reshaped = x_embed.view(max_sent_length * batch_size, max_word_length, char_embed_size).permute(0, 2, 1)

        x_conv_out = self.cnn.forward(x_reshaped)
        x_highway = self.highway.forward(x_conv_out)
        x_word_emb = self.dropout.forward(x_highway)

        output = x_word_emb.view(max_sent_length, batch_size, -1)
        return output



        ### END YOUR CODE

