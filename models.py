from treelstm import TreeLSTM
import torch
from torch import nn
from data_handler import sent_to_tree, read_iob


class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(DenseNet, self).__init__()
        self.hidden_size = hidden_size 
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.device = device

    def prepare_input(self, sent, featurizer):
        features = featurizer(sent)
        input = torch.tensor(features, dtype=torch.float, device=self.device)
        return input

    def forward(self, input):
        dense_out = self.activation(self.dense(input))
        classifier_out = self.classifier(dense_out)
        output = self.softmax(classifier_out)
        return output

class TreeLSTMParser(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(TreeLSTMParser, self).__init__()
        self.tree = TreeLSTM(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.device = device

    def prepare_input(self, sent, featurizer):
        features = featurizer(sent)
        features_tensor = torch.tensor(features, device=self.device).float()
        labels = read_iob(sent, self.device)
        data = sent_to_tree(sent, features, labels, self.device)  
        return (data['features'], data['node_order'], data['adjacency_list'], data['edge_order'], features_tensor)

    def fake_batch(self, input):
        return input.reshape(len(input),1,-1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device))

    def forward(self, input):
        tree_features, node_order, adjacencies, edge_order, features_tensor = input
        clean_hidden = self.init_hidden()
        lstm_in = self.fake_batch(features_tensor)
        
        lstm_hidden, lstm_cell = self.lstm(lstm_in, clean_hidden)
        tree_hidden, tree_cell = self.tree(tree_features, node_order, adjacencies, edge_order)
        lstm_out = lstm_hidden.reshape(len(lstm_hidden), -1)
        classifier_in = torch.cat([tree_hidden, lstm_out], 1)
        classifier_out = self.classifier(classifier_in)
        out = self.softmax(classifier_out)
        return out



class TreeBiLSTMParser(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(TreeBiLSTMParser, self).__init__()
        self.tree = TreeLSTM(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 3, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.device = device

    def prepare_input(self, sent, featurizer):
        features = featurizer(sent)
        features_tensor = torch.tensor(features, device=self.device).float()
        labels = read_iob(sent, self.device)
        data = sent_to_tree(sent, features, labels, self.device)  
        return (data['features'], data['node_order'], data['adjacency_list'], data['edge_order'], features_tensor)

    def fake_batch(self, input):
        return input.reshape(len(input),1,-1)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_size, device=self.device),
                torch.zeros(2, 1, self.hidden_size, device=self.device))

    def forward(self, input):
        tree_features, node_order, adjacencies, edge_order, features_tensor = input
        clean_hidden = self.init_hidden()
        lstm_in = self.fake_batch(features_tensor)
        
        lstm_hidden, lstm_cell = self.lstm(lstm_in, clean_hidden)
        tree_hidden, tree_cell = self.tree(tree_features, node_order, adjacencies, edge_order)
        lstm_out = lstm_hidden.reshape(len(lstm_hidden), -1)
        classifier_in = torch.cat([tree_hidden, lstm_out], 1)
        classifier_out = self.classifier(classifier_in)
        out = self.softmax(classifier_out)
        return out




class LSTMParser(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(LSTMParser, self).__init__()
        self.hidden_size = hidden_size 
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.device = device

    def prepare_input(self, sent, featurizer):
        features = featurizer(sent)
        input = torch.tensor(features, dtype=torch.float, device=self.device)
        input = self.fake_batch(input)
        return input

    def fake_batch(self, input):
        return input.reshape(len(input),1,-1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device))

    def forward(self, input):
        # making a batch of 1 sequence out of a sequence
        clean_hidden = self.init_hidden()
        lstm_out, _ = self.lstm(input, clean_hidden)#, clean_hidden)
        reshaped = lstm_out.reshape(len(input), -1)
        tagger_out = self.hidden(reshaped)
        out = self.softmax(tagger_out) # dim=1?
        return out


class BiLSTMParser(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(BiLSTMParser, self).__init__()
        self.hidden_size = hidden_size 
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.hidden = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.device = device

    def prepare_input(self, sent, featurizer):
        features = featurizer(sent)
        input = torch.tensor(features, dtype=torch.float, device=self.device) # device
        input = self.fake_batch(input)
        return input

    def fake_batch(self, input):
        return input.reshape(len(input),1,-1)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_size, device=self.device),
                torch.zeros(2, 1, self.hidden_size, device=self.device))

    def forward(self, input):
        clean_hidden = self.init_hidden()
        lstm_out, _ = self.lstm(input, clean_hidden)
        reshaped = lstm_out.reshape(len(input), -1)
        tagger_out = self.hidden(reshaped)
        out = self.softmax(tagger_out) # dim=1?
        return out

