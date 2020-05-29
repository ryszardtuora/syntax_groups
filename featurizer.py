import numpy
from data_handler import *

class W2V_Featurizer():
    def __init__(self, emb, device):
        self.emb = emb
        self.input_dim = 300
        self.device = device
    

    def __call__(self, sent):
        features = read_words(sent, self.emb, self.device)
        return features


class W2V_POS_Featurizer():
    def __init__(self, emb, device):
        self.emb = emb
        self.input_dim = 337
        self.device = device


    def __call__(self, sent):
        words = read_words(sent, self.emb, self.device)
        pos = read_pos_tags(sent, self.device)
        features = [numpy.concatenate([w, p]) for w, p in zip(words,pos)]
        return features


class W2V_POS_DEPREL_Featurizer():
    def __init__(self, emb, device):
        self.emb = emb
        self.input_dim = 369
        self.device = device


    def __call__(self, sent):
        words = read_words(sent, self.emb, self.device)
        pos = read_pos_tags(sent, self.device)
        deprels = read_deprels(sent, self.device)
        features = [numpy.concatenate([w, p, d]) for w, p, d in zip(words, pos, deprels)]
        return features

