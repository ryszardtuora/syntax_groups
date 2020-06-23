import gensim
import conllu
import numpy
import torch
import random
import sys
from torch import nn
from tqdm import tqdm

from models import DenseNet, BiLSTMParser, LSTMParser, TreeLSTMParser, TreeBiLSTMParser
from data_handler import *
from evaluation import fix, compare, test_sent, analyse_output
from featurizer import W2V_Featurizer, W2V_POS_Featurizer, W2V_POS_DEPREL_Featurizer

IOB = "IOB"
embeddings_path = "cut.vec"
num_classes = 3
device = torch.device("cpu")
model_name = "BiLSTMParser.model"


def out_to_iob(out):
  iob = ""
  for y in out:
    topv, topi = y.topk(1)
    topi = topi.item()
    iob_ann = "IOB"[topi]
    iob += iob_ann
  return iob


def load_model():
  emb = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
  featurizer = W2V_POS_DEPREL_Featurizer(emb, device)
  input_dim = featurizer.input_dim
  model = BiLSTMParser(input_dim, 100, 3, device)
  state_dict = torch.load(model_name)
  model.load_state_dict(state_dict)
  model.eval()
  return model, featurizer


def process_sent(model, featurizer, sent):
  with torch.no_grad():
    model_input = model.prepare_input(sent, featurizer)
    output = model(model_input)
  iob_out = out_to_iob(output)
  fixed_iob = fix(iob_out)
  return fixed_iob


def load_sents(file):
  with open(file) as f:
    txt = f.read()
  sents = conllu.parse(txt)
  return sents

# Example of usage
# model, featurizer = load_model()
# sents = load_sents(file) # .conllu file annotated with tags and dependency relations
# example_sent = sents[-1]
# output = process_sent(model, featurizer, example_sent)
# # printing results
# for ind, tok, ann in zip(range(len(example_sent)), example_sent, output):
#   print("{0:4} {1:20} {2:4}".format(ind, tok["form"], ann))
