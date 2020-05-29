import conllu
import os
from lxml import etree
from nkjp_reader import get_nodes_from_file, HEADER, W3HEADER
from functools import reduce

from spacy import displacy
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span
from conllu import parse_tree_incr, parse_incr, parse, parse_tree
from conllu.models import TokenList

FILE_PATH = "PDBUD-master/NKJP1M-UD_current/NKJP1M-UD.conllu"
NKJP_DIR = "NKJP"
TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
W3_NAMESPACE = "http://www.w3.org/XML/1998/namespace"

dummy_voc = Vocab()

def split_origin(origin_name):
  name, sent_number = origin_name.split("#")
  split_name = name.split("_", 1)
  sample_name = split_name[0]
  sent_id = split_name[1]
  return sample_name, sent_id, sent_number


def build_sample_dictionary(sents):
  # sample to -> (index, morph_sent id) dict
  sample_dictionary = {}
  for i, sent in enumerate(sents):
    origin = sent.metadata["orig_file_sentence"]
    sample_name, sent_id, sent_number = split_origin(origin)
    try:
      sample_dictionary[sample_name].append((i, sent_id))
    except KeyError:
      sample_dictionary[sample_name]=[(i, sent_id)]
  return sample_dictionary


def process_sample(sample_path):
  # Buliding the group_sentence -> word_sentence id dict
  group_file = os.path.join(sample_path, "ann_groups.xml")
  if not os.path.exists(group_file):
    return False, False

  group_et = etree.parse(group_file)
  group_sents = group_et.xpath(".//x:s", namespaces={"x": TEI_NAMESPACE})
  group_sent_ids = {s:s.get("corresp").split("#")[1] for s in group_sents}

  # Building the word_sentence id -> morphosyntax_sentence id dict
  words_et = etree.parse(os.path.join(sample_path, "ann_words.xml"))
  word_sents = words_et.xpath(".//x:s", namespaces={"x": TEI_NAMESPACE})
  word_sent_ids = {ws.xpath("./@x:id", namespaces={"x": W3_NAMESPACE})[0]: ws.get("corresp").split("#")[1] for ws in word_sents}
  # Building the word -> [morphosyntactic segment ids] dict
  words = words_et.xpath(".//x:seg", namespaces={"x": TEI_NAMESPACE})
  word_dic = {}
  for word in words:
    ptrs = word.xpath(".//x:ptr", namespaces={"x": TEI_NAMESPACE})
    ptrs_split = [p.get("target").split("#") for p in ptrs]
    ptr_targets = [p[1] for p in ptrs_split if len(p) == 2]
    word_id = word.xpath("./@x:id", namespaces={"x": W3_NAMESPACE})[0]
    word_dic[word_id] = ptr_targets

  # Building the morphosyntax_sent_id -> (morphosyntactic_seg_id -> index dict) dict
  morph_et = etree.parse(os.path.join(sample_path, "ann_morphosyntax.xml"))
  morph_sents = morph_et.xpath(".//x:s", namespaces={"x": TEI_NAMESPACE})
  morph_sent_table = {}
  for morph_sent in morph_sents:
    morph_sent_id = morph_sent.xpath("./@x:id", namespaces={"x": W3_NAMESPACE})[0]
    morph_sent_toks = morph_sent.xpath(".//x:seg", namespaces={"x": TEI_NAMESPACE})
    morph_sent_tok_dic = {}
    for i, tok in enumerate(morph_sent_toks):
      morph_sent_tok_id = tok.xpath("./@x:id", namespaces={"x": W3_NAMESPACE})[0]
      morph_sent_tok_dic[morph_sent_tok_id] = i
    morph_sent_table[morph_sent_id] = morph_sent_tok_dic

  # Building a morphosyntax_sent id -> [groups] dictionary
  sent_to_groups = {}
  for group_sent, sent_id in group_sent_ids.items():
    morph_sent_id = word_sent_ids[sent_id]
    group_sent_ids[group_sent] = morph_sent_id
    sent_to_groups[group_sent_ids[group_sent]] = []
    groups = group_sent.xpath(".//x:seg", namespaces={"x": TEI_NAMESPACE})
    for group in groups:
      get_ind = lambda t : morph_sent_table[morph_sent_id][t]
      #print(morph_sent_table[morph_sent_id])
      ptrs_split = [p.get("target").split("#") for p in group.xpath(".//x:ptr", namespaces={"x": TEI_NAMESPACE})]
      ptrs_targets = [p[1] for p in ptrs_split if len(p)==2]
      morph_target_lists = [word_dic[t] for t in ptrs_targets]
      
      if morph_target_lists == []:
        # this might get triggered when a group which is identical to other group in 
        # terms of span is processed
        continue
      
      morph_targets = reduce(lambda l1, l2: l1+l2, morph_target_lists)# flattening the pointers list
      morph_targets = [get_ind(m) for m in morph_targets]
      start = morph_targets[0]
      end = morph_targets[-1]

      group_type_el = group.xpath(".//x:f[@name='type']", namespaces={"x": TEI_NAMESPACE})[0]
      group_type = group_type_el.xpath(".//x:symbol", namespaces={"x": TEI_NAMESPACE})[0].get("value")

      orth_el = group.xpath(".//x:f[@name='orth']", namespaces={"x": TEI_NAMESPACE})[0]
      orth = orth_el.xpath(".//x:string", namespaces={"x": TEI_NAMESPACE})[0].text

      semh_el = group.xpath(".//x:f[@name='semh']", namespaces={"x": TEI_NAMESPACE})[0]
      semh = [get_ind(s) for s in word_dic[semh_el.get("fVal").split("#")[1]]] # this may evaluate to more than one morpohsyntactic segment
      synh_el = group.xpath(".//x:f[@name='synh']", namespaces={"x": TEI_NAMESPACE})[0]
      synh = [get_ind(s) for s in word_dic[synh_el.get("fVal").split("#")[1]]]

      group_entry = {"type": group_type,
                     "orth": orth,
                     "semh": semh,
                     "synh": synh,
                     "morph_indices": morph_targets,
                     "start": start,
                     "end": end}
      sent_to_groups[group_sent_ids[group_sent]].append(group_entry)
  return sent_to_groups, morph_sent_table

def plot_phrases(sentence, groups):
  words = [t["form"] for t in sentence]
  doc = Doc(dummy_voc, words)
  ents = []
  for group in groups:
    label = group["type"]
    start = group["morph_indices"][0]
    end = group["morph_indices"][-1]+1
    ents.append(Span(doc, start, end, label=label))
  doc.ents = ents
  plot = displacy.render(doc, style="ent", page=True)
  return plot

def remove_subgroups(groups):
  to_remove = []
  for i, g1 in enumerate(groups):
    for g2 in groups[:i] + groups [i+1:]:
      if g1["start"] >= g2["start"] and g1["end"] <= g2["end"] and not(g1["start"] == g2["start"] and g1["end"] == g2["end"]): 
        to_remove.append(g1)
  clean = [x for x in groups if x not in to_remove]
  return clean

def groups_to_iob(sent, group_list):
  # removing spanning tokens
  to_remove = []
  for t in sent:
    if type(t["id"]) == tuple:
      to_remove.append(t)
  for t in to_remove:
    sent.remove(t)

  iob = ["O"] * len(sent)
  for g in group_list:
    iob[g["start"]] = "B"
    for i in range(g["start"]+1, g["end"]+1):
      iob[i] = "I"
  compact_iob = "".join(iob)
  #print(iob)
  return compact_iob


def main():
  f = open(FILE_PATH, "r", encoding="utf-8")
  txt = f.read()
  sents = conllu.parse(txt)
  f.close()
  
  sample_dic = build_sample_dictionary(sents)
  #print(sample_dic)
  iob_out = []
  for sample_name in list(sample_dic.keys()):
    sent_to_groups, morph_sent_table = process_sample("NKJP/"+sample_name)
    if not sent_to_groups: 
      # no groups defined for this sample
      continue
    for i, sent_id in sample_dic[sample_name]:
      sent = sents[i]
      if sent.metadata["PDB"] != "False":
        # PDB sentences might differ in segmentation
        continue
      groups = sent_to_groups[sent_id]
      clean_groups = remove_subgroups(groups)
      try:
        iob = groups_to_iob(sent, clean_groups)
        line = "\t".join(["{}#{}".format(sample_name, sent_id), iob])
        iob_out.append(line)
      except IndexError:
        pass#return sent_to_groups, morph_sent_table, sent
  iob_txt = "\n".join(iob_out)
  with open("NKJP.iob", "w", encoding="utf-8") as f:
    f.write(iob_txt)


main()


# we assume that sent ids in groups match ids elsewhere
