import re
import numpy

ENT_REGEX = re.compile(r"BI*")

def out_to_iob(model_out):
  iob = ""
  for o in model_out:
    iob += "IOB"[numpy.argmax(o)]
  return iob


def fix(model_iob):
  prev = None
  fixed = []
  for tok in model_iob:
    if tok == "I" and prev not in ["B", "I"]:
      fixed.append("B")
    else:
      fixed.append(tok)
    prev = tok
  fixed_iob = "".join(fixed)
  return fixed_iob


def compare(gold_iob, model_iob):
  gold_ents = list(ENT_REGEX.finditer(gold_iob))
  gold_spans = set([e.span() for e in gold_ents])

  model_ents = list(ENT_REGEX.finditer(model_iob))
  model_spans = set([e.span() for e in model_ents])

  TP = len(gold_spans.intersection(model_spans))
  FP = len(model_spans.difference(gold_spans))
  FN = len(gold_spans.difference(model_spans))

  return TP, FP, FN


