
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# In[3]:

class biLM(nn.Module):
    '''
    initialize with
    embedding: pre-trained embedding layer
    hidden_size: size of hidden_states of biLM
    n_layers: number of layers
    dropout: dropout
    '''
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(biLM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding
        self.forwardLSTM = nn.LSTM(hidden_size, 
                                         hidden_size, 
                                         n_layers, 
                                         dropout=(0 if n_layers == 1 else dropout))
        self.backwardLSTM = nn.LSTM(hidden_size, 
                                         hidden_size, 
                                         n_layers, 
                                         dropout=(0 if n_layers == 1 else dropout))
        
    def forward(self, input_seq, input_lengths, initial_states=None):
        '''
        input_seq: size=(MAX_LEN, batch_size)
        input_lengths: contains length of each sentence
        initial_states: tuple of initial hidden_state of LSTM, initial cell state of LSTM
        '''
        embedded = self.embedding(input_seq)
        MAX_LEN = embedded.size()[0]
        # embedded: size=(MAX_LEN, batch_size, hidden_size)
        outputs = torch.zeros(MAX_LEN, batch_size, 2, hidden_size)
        hidden_states = torch.zeros(self.n_layers * 2, MAX_LEN, batch_size, hidden_size)
        batch_size = embedded.size()[1]
        
        for batch_n in batch_size:
            sentence = input_seq[:,batch_n, :]
            length = input_lengths[batch_n]
            
            if initial_states:
                hidden_forward_state, cell_forward_state = initial_states
                hidden_backward_state, cell_backward_state = initial_states
            else:
                hidden_forward_state, cell_forward_state = None, None
                hidden_backward_state, cell_backward_state = None, None
                
            for t in range(length):
                output, (hidden_forward_state, cell_forward_state) = forwardLSTM(sentence[t], (hidden_forward_state, cell_forward_state))
                outputs[t, batch_n, 0, :] = output[0, 0, :]
                hidden_state[:n_layers, batch_n, t, :] = hidden_forward_state[:, 0, :]
                
            for t in range(length):
                output, (hidden_backward_state, cell_backward_state) = backwardLSTM(sentence[length - t - 1], (hidden_backward_state, cell_backward_state))
                outputs[length - t - 1, batch_n, 1, :] = output[0, 0, :]
                hidden_state[n_layers:, batch_n, length - t - 1, :] = hidden_backward_state[:, 0, :]
                
        return outputs, hidden_states, embedded


# In[4]:

class ELMo(nn.Module):
    '''
    initialize with
    
    '''
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, l2_coef=None, do_layer_norm=False):
        super(ELMo, self).__init__()
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.use_top_only = use_top_only
        self.do_layer_norm = do_layer_norm
        self.biLM = biLM(hidden_size, embedding, n_layers, dropout)
        self.W = torch.zeros(2*n_layers+1, requires_grad=True)
        self.W = F.softmax(self.W + 1/(2*n_layers + 1))
        self.gamma = torch.ones(1, requires_grad=True)
        
    def do_norm(layer, mask):
        masked_layer = layer * mask
        N = torch.sum(mask) * self.hidden_size
        mean = torch.sum(masked_layer)/N
        variance = torch.sum(((masked_layer - mean) * mask) ** 2) / N
        
        return F.batch_norm(layer, mean, variance)
    
    def forward(self, input_seq, input_lengths, mask, initial_states=None):
        bilm_outputs, hidden_states, embedded = biLM(input_seq, input_lengths, initial_states)
        concat_hidden_with_embedding = torch.cat(embedded.unsqueeze(0), hidden_states)
        ELMo_embedding = gamma.item()
        for i in range(2*n_layers + 1):
            w = W[i]
            layer = concat_hidden_with_embedding[i]
            if do_layer_norm:
                layer = do_norm(layer, mask)
            ELMo_embedding = ELMo_embedding + w * layer
        return ELMo_embedding, bilm_outputs


# In[ ]:



