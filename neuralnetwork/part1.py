#!/usr/bin/env python3

## Written by Lei Xie
## Neural Networks and Deep Learning practice


import torch

class rnn(torch.nn.Module):

    def __init__(self):
        super(rnn, self).__init__()

        self.ih = torch.nn.Linear(64, 128)
        self.hh = torch.nn.Linear(128, 128)

    def rnnCell(self, input, hidden):
        ## The network should take some input (inputDim = 64) and the current 
        ## hidden state (hiddenDim = 128), and return the new hidden state.
        x = torch.add(self.ih(input), self.hh(hidden))
        return torch.tanh(x)        

    def forward(self, input):
        hidden = torch.zeros(128)

        ## Return the final hidden state after the last input in the sequence has been processed.

        for each_seq in input:
            hidden = self.rnnCell(each_seq, hidden)
        return hidden


class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        ## network defined by this class is equivalent to the one defined in class "rnn".
        self.net = torch.nn.RNN(64, 128)

    def forward(self, input):
        _, hidden = self.net(input)

        return hidden

def lstm(input, hiddenSize):

    ## Variable input is of size [batchSize, seqLength, inputDim]

    lstm = torch.nn.LSTM(input.shape[2], hiddenSize, batch_first = True)
    return lstm(input)

def conv(input, weight):

    ## The convolution should be along the sequence axis. Input is of size [batchSize, inputDim, seqLength]

    out_channels, in_groups, kernel_size = weight.shape
    in_channels = input.shape[1]
    conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias = False)
    conv.weight.data = weight
    return conv(input)

    # conv1d = torch.nn.funcational.conv1d(input, weight)
    # return conv1d(input)