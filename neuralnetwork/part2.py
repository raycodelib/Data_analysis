#!/usr/bin/env python3

## Written by Lei Xie 
## Neural Networks and Deep Learning practice2

import numpy as np

import torch
import torch.nn as tnn
import torch.optim as topti

from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class NetworkLstm(tnn.Module):
    """
    the LSTM-based network accepts batched 50-d
    vectorized inputs, with the following structure:
    LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
    """

    def __init__(self):
        super(NetworkLstm, self).__init__()

        ## Create and initialise weights and biases for the layers.
        self.LSTM = tnn.LSTM(50, 100, batch_first=True)
        self.Linear_64 = tnn.Linear(100, 64)
        self.ReLu = tnn.ReLU()
        self.Linear_1 = tnn.Linear(64, 1)
 
    def forward(self, input, length):

        ## Create the forward pass through the network.

        x, _ = self.LSTM(input)
        x = self.Linear_64(x[np.arange(input.shape[0]), length - 1])
        x = self.ReLu(x)
        x = self.Linear_1(x)
        x = x.view(-1)

        return x

# Class for creating the neural network.
class NetworkCnn(tnn.Module):
    """
    All conv layers should be of the form:
    conv1d(channels=50, kernel size=8, padding=5)
    Conv -> ReLu -> maxpool(size=4) -> Conv -> ReLu -> maxpool(size=4) ->
    Conv -> ReLu -> maxpool over time (global pooling) -> Linear(1)
    The max pool over time operation refers to taking the
    maximum val from the entire output channel. See Kim et. al. 2014:
    https://www.aclweb.org/anthology/D14-1181/
    """

    def __init__(self):
        super(NetworkCnn, self).__init__()

        ## Create and initialise weights and biases for the layers.

        self.conv = tnn.Conv1d(50, 50, 8, padding=5)
        self.ReLu = tnn.ReLU()
        self.maxpool = tnn.MaxPool1d(4)
        self.Linear_1 = tnn.Linear(50, 1)
        
    def forward(self, input, length):

        ## Create the forward pass through the network.
        input = input.permute(0, 2, 1)
        x = self.conv(input)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.ReLu(x)
        x = x.max(dim=-1)[0] 
        x = self.Linear_1(x)
        x = x.view(-1)
        
        return x

def lossFunc():
    ##  the loss to add a sigmoid to the output and calculate the binary     cross-entropy.
    return tnn.BCEWithLogitsLoss()

def measures(outputs, labels):
    """
    Return the number of true positive classifications, 
    true negatives, false positives and false, 
    negatives from the given batch outputs and provided labels.
    """
    true_positive  = torch.sum((outputs >0) & (labels==1)).item()
    true_negative  = torch.sum((outputs <0) & (labels==0)).item()
    false_positive = torch.sum((outputs >0) & (labels==0)).item()
    false_negative = torch.sum((outputs <0) & (labels==1)).item()
         
    return true_positive, true_negative, false_positive, false_negative


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    from imdb_dataloader import IMDB
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    # Create an instance of the network in memory (potentially GPU memory). Can change to NetworkCnn during development.
    # net = NetworkLstm().to(device)
    net = NetworkCnn().to(device)

    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()
