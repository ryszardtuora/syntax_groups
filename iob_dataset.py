import os
import torch
import conllu
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy
from tree import Tree
import Constants


IOB = "IOB"

class IOBDataset(data.Dataset):
  def __init__(self, path, vocab):
    super(IOBDataset, self).__init__()
    self.vocab = vocab
    self.num_classes = 3
    
    self.sentences = []
    self.trees = []
    self.iob = []

    data = self.read_input(path)    
    self.size = len(self.sentences)

  def __len__(self):
    return self.size    


  def __getitem__(self, index):
    sentence = deepcopy(self.sentences[index])
    tree = deepcopy(self.trees[index])
    iob = deepcopy(self.iob[index])
    return (sentence, tree, iob)
    
    
  def read_input(self, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
      for s in tqdm(conllu.parse_incr(f)):
        self.sentences.append(self.read_sentence(s))
        self.trees.append(self.read_tree(s))
        self.iob.append(self.read_iob(s))


  def read_sentence(self, sent):
    forms = [tok["form"] for tok in sent]
    indices = self.vocab.convertToIdx(forms, Constants.UNK_WORD)
    sent_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
    return sent_tensor


  def read_iob(self, sent):
    iob_ann = [IOB.index(tok["misc"]["iob"]) for tok in sent]
    iob_tensor = torch.tensor(iob_ann, dtype=torch.long, device="cpu")
    return iob_tensor


  def read_tree(self, sent):
    # trees are indexed 0..n-1 where n is tree span
    parents = [tok["head"] for tok in sent]
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
      if i - 1 not in trees.keys() and parents[i - 1] != -1:
          idx = i
          prev = None
          while True:
            parent = parents[idx - 1]
            if parent == -1:
              break
            tree = Tree()
            if prev is not None:
              tree.add_child(prev)
            trees[idx - 1] = tree
            tree.idx = idx - 1
            if parent - 1 in trees.keys():
              trees[parent - 1].add_child(tree)
              break
            elif parent == 0:
              root = tree
              break
            else:
              prev = tree
              idx = parent
    return root
