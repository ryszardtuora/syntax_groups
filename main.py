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
hidden_dim = 100
tree_size = 100
lstm_size = 100
embeddings_path = "cut.vec"
epochs = 15
num_classes = 3
TRAIN_PATH = "phrases_train.conllu"
TEST_PATH = "phrases_test.conllu"
DEV_PATH = "phrases_dev.conllu"
test_path = "phrases_test.conllu"
device = torch.device("cpu")


def out_to_one_hot(indices):
  y = torch.zeros(len(indices), num_classes, dtype=torch.long, device=device)
  for n, i in enumerate(indices):
    y[n][i] = 1
  return y


def train_sent(model, sent, featurizer, criterion, optimizer):
    model_input = model.prepare_input(sent, featurizer)
    labels = read_iob(sent, device)
    
    loss = 0
    model.zero_grad()
    output = model(model_input).float()
    loss = criterion(output, labels)
    loss.backward()
    
    optimizer.step()
    return float(loss)

def print_example(model, sent, featurizer):
    model_input = model.prepare_input(sent, featurizer)
    labels = read_iob(sent, device)
    with torch.no_grad():
        out = model(model_input)
        print("sys ", fix(out_to_iob(out)))
        print("gold ", "".join(["IOB"[i] for i in labels]), "\n")


def out_to_iob(out):
  iob = ""
  for y in out:
    topv, topi = y.topk(1)
    topi = topi.item()
    iob_ann = "IOB"[topi]
    iob += iob_ann
  return iob


def main():
    emb = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    #emb = {}
    featurizer = W2V_POS_DEPREL_Featurizer(emb, device)
    input_dim = featurizer.input_dim
    criterion = nn.CrossEntropyLoss()
    model = BiLSTMParser(input_dim, 100, 3, device)
    optimizer = torch.optim.Adam(model.parameters())

    train_sentences = load_conllu(TRAIN_PATH)
    dev_sentences = load_conllu(DEV_PATH)
    test_sentences = load_conllu(TEST_PATH)


    top_f1 = -float("inf")
    for e in range(epochs):
        model.train()
        total_loss = 0
        train_indices = list(range(len(train_sentences)))
        random.shuffle(train_indices)
        for i in tqdm(train_indices):
            s = train_sentences[i]
            total_loss += train_sent(model, s, featurizer, criterion, optimizer)
        print("Train loss: ", total_loss)

        model.eval()
        corr = 0
        total = 0
        for sent in dev_sentences[-10:]:
            print_example(model, sent, featurizer)

        TP, FP, FN, SENTS = (0, 0, 0, 0)
        no_sents = 0
        dev_loss = 0
        for s in tqdm(dev_sentences):
            ncorr, ntotal, iob_out, loss = test_sent(model, s, featurizer, criterion)
            dev_loss += loss
            corr += ncorr
            total += ntotal
            fixed_iob = fix(iob_out)
            iob_ann = "".join([tok["misc"]["iob"] for tok in s])
            comparison = compare(iob_ann, fixed_iob)
            TP += comparison[0]
            FP += comparison[1]
            FN += comparison[2]
            SENTS += int(fixed_iob == iob_ann)
            no_sents += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        sent_acc = SENTS / no_sents
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("sents_acc: ", sent_acc)
        print("Dev loss: ", dev_loss)
        print(corr/total)
        if f1 > top_f1:
            top_f1 = f1
            torch.save(model.state_dict(), type(model).__name__ + ".model")

    # test set evaluation
    state_dict = torch.load(type(model).__name__ + ".model")
    model.load_state_dict(state_dict)
    model.eval()
    corr = 0
    total = 0
    TP, FP, FN, SENTS = (0, 0, 0, 0)
    no_sents = 0
    for s in tqdm(test_sentences):
        ncorr, ntotal, iob_out, loss = test_sent(model, s, featurizer, criterion)
        corr += ncorr
        total += ntotal
        fixed_iob = fix(iob_out)
        iob_ann = "".join([tok["misc"]["iob"] for tok in s])
        comparison = compare(iob_ann, fixed_iob)
        TP += comparison[0]
        FP += comparison[1]
        FN += comparison[2]
        SENTS += int(fixed_iob == iob_ann)
        no_sents += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    sent_acc = SENTS / no_sents
    print("\n\n TEST SET\n\n")
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("sents_acc: ", sent_acc)
    print(corr/total)



#main()




