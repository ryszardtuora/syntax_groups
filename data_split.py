import conllu
import random
from collections import OrderedDict


CONLLU_PATH = "PDBUD-master/NKJP1M-UD_current/NKJP1M-UD.conllu"
IOB_PATH = "NKJP.iob"

def split_origin(origin_name):
  name, sent_number = origin_name.split("#")
  split_name = name.split("_", 1)
  sample_name = split_name[0]
  sent_id = split_name[1]
  return sample_name, sent_id, sent_number


def load_conllu(conllu_path):
  with open(conllu_path, "r", encoding="utf-8") as f:
    txt = f.read()
  sents = conllu.parse(txt)

  sent_dict = {}
  for sent in sents:
    remove_span_tokens(sent)
    origin = sent.metadata["orig_file_sentence"]
    sample_name, sent_id, sent_number = split_origin(origin)
    sent_dict[sample_name + "#" + sent_id] = sent

  return sent_dict


def load_iob(iob_path):
  with open(iob_path, "r", encoding="utf-8") as f:
    txt = f.read()
  lines = txt.split("\n")
  
  iob_dict = {}
  for l in lines:
    sent_id, iob = l.split("\t")
    iob_dict[sent_id] = iob
  
  return iob_dict


def remove_span_tokens(sent):
  # removing spanning tokens
  to_remove = []
  for t in sent:
    if type(t["id"]) == tuple:
      to_remove.append(t)
  for t in to_remove:
    sent.remove(t)


def generate_data(conllu_path, iob_path):
  con = load_conllu(conllu_path)
  iob = load_iob(iob_path)
  indices = list(iob.keys())

  for i in indices:
    sent = con[i]
    iob_ann = iob[i]
    for tok, iob_ann in zip(sent, iob_ann):    
      misc = tok["misc"]
      if misc:
        misc["iob"] = iob_ann
      else:
        tok["misc"] = OrderedDict({"iob": iob_ann})

  random.shuffle(indices)
  train = indices[:-4000]
  dev = indices[-4000:-2000]
  test = indices[-2000:]
  
  train_string = "\n\n".join([con[i].serialize() for i in train]) + "\n"
  dev_string = "\n\n".join([con[i].serialize() for i in dev]) + "\n"
  test_string = "\n\n".join([con[i].serialize() for i in test]) + "\n"

  with open("phrases_train.conllu", "w", encoding="utf-8") as f:
    f.write(train_string)

  with open("phrases_dev.conllu", "w", encoding="utf-8") as f:
    f.write(dev_string)

  with open("phrases_test.conllu", "w", encoding="utf-8") as f:
    f.write(test_string)

generate_data(CONLLU_PATH, IOB_PATH)
