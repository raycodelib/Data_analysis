#!/usr/bin/env python3

## Written by Lei Xie 
## Neural Networks and Deep Learning practice3

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # LSTM model
        # self.LSTM = tnn.LSTM(50, 100, batch_first=True)
        # self.Linear_64 = tnn.Linear(100, 64)
        # self.ReLu = tnn.ReLU()
        # self.Linear_1 = tnn.Linear(64, 1)

        # CNN model
        # self.conv = tnn.Conv1d(50, 50, 8, padding=5)
        # self.ReLu = tnn.ReLU()
        # self.maxpool = tnn.MaxPool1d(4)
        # self.Linear_1 = tnn.Linear(50, 1)

        # GRU model
        self.GRU = tnn.GRU(50, 100, batch_first=True)
        self.Linear_64 = tnn.Linear(100, 64)
        self.ReLu = tnn.ReLU()
        self.Linear_1 = tnn.Linear(64, 1)
        

    def forward(self, input, length):
        # LSTM model 
        # x, _ = self.LSTM(input)
        # x = self.Linear_64(x[np.arange(input.shape[0]), length - 1])
        # x = self.ReLu(x)
        # x = self.Linear_1(x)
        # x = x.squeeze()
        # x = x.view(-1)
        # return x

        # CNN model
        # input = input.permute(0, 2, 1)
        # x = self.conv(input)
        # x = self.ReLu(x)
        # x = self.maxpool(x)
        # x = self.conv(x)
        # x = self.ReLu(x)
        # x = self.maxpool(x)
        # x = self.conv(x)
        # x = self.ReLu(x)
        # x = x.max(dim=-1)[0]  
        # x = self.Linear_1(x)
        # x = x.view(-1)
        # return x

        #GRU model
        x, _ = self.GRU(input)
        x = self.Linear_64(x[np.arange(input.shape[0]), length - 1])
        x = self.ReLu(x)
        x = self.Linear_1(x)
        # x = x.view(-1)
        x = x.squeeze()
        return x
        
class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        new_x = []
        for word in x:
            # remove irrelevant symbols, e.g.  'Bob.' ---> 'Bob'
            word = word.replace('.', '')
            word = word.replace(',', '')
            word = word.replace('/>', '')
            word = word.replace('</', '')
            word = word.replace('>', '')
            word = word.replace('<', '')
            word = word.replace('!', '')
            word = word.replace(':', '')
            word = word.replace(';', '')
            word = word.replace('"', '')
            word = word.replace('"', '')
            word = word.replace('^', '')
            word = word.replace('*', '')
            word = word.replace('(', '')
            word = word.replace(')', '')
            word = word.replace('---', '')
            word = word.replace('--', '')
            word = word.replace('?', '')
            word = word.replace('@', '')
            word = word.replace('&', '')
            # word = word.replace('', '')
            new_x.append(word)
            # print(word)

        # remove meaningless stopwords, the list is copied from nltk stopwords set
        stop_words = {'any', 'such', "wasn't", 'haven', 'other', "it's", 'a', 'just', 'couldn', "you're", 
        'off', 'ourselves', 'myself', 'or', 'weren', 'didn', 'this', 'shan', 'into', 're', 'why', 'hasn', 
        'of', 'doesn', 'am', 'there', 'her', 'at', 'no', "hadn't", 'again', "needn't", 'during', 'very', 
        'themselves', 'needn', 'my', 'won', 'its', 'can', "you'd", 'against', 'being', 'that', 'ma', 'then', 
        'above', 'i', 'own', 'from', 'more', 'm', 'for', 'yours', 'who', 'theirs', 'in', 'than', 'isn', 
        'himself', 'nor', 'his', 'him', 'she', 't', "don't", "doesn't", 'mightn', 'our', 'had', 'don', 
        'herself', 'they', "couldn't", 'and', 'wouldn', 'y', 'mustn', 'both', 'so', 'each', 'your', 'too', 
        'their', 'if', 'doing', "she's", 'once', 'having', 'when', "wouldn't", "hasn't", 'with', 'because', 
        "shan't", "shouldn't", "that'll", 'before', 'how', 'which', "weren't", 'out', 'on', "isn't", 'where', 
        'after', 'hadn', 'to', 'now', 'is', 'an', 'through', 'between', 've', 'do', 'over', 'it', 'about', 
        "won't", 'was', "mightn't", 'down', 'o', "you'll", 'whom', 'ours', "didn't", 'same', 's', 'those', 
        'as', "you've", 'shouldn', 'some', 'will', 'did', 'these', 'yourselves', 'but', 'by', 'few', 'all', 
        'aren', 'yourself', 'until', 'further', 'be', 'have', 'does', 'only', "mustn't", 'me', "haven't", 
        'under', 'hers', "aren't", 'itself', 'were', 'up', 'wasn', 'below', 'been', 'most', 'should', 'not', 
        "should've", 'll', 'them', 'you', 'has', 'what', 'we', 'while', 'he', 'the', 'here', 'd', 'are', 'ain', '.'}
        
        x = [w for w in new_x if not w in stop_words]
        # print('after stop words filter', x)
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""


        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
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

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
