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


def test_sent(model, sent, featurizer, criterion):
    labels = read_iob(sent, device)      
    corr = 0
    IOBs = "".join([IOB[ind.item()] for ind in labels])
    with torch.no_grad():
        model_input = model.prepare_input(sent, featurizer)
        output = model(model_input)
        iob_out = out_to_iob(output)
        for sys,gold in zip(iob_out, IOBs):
            if sys == gold:
                corr +=1

    loss = criterion(output.float(), labels)

    return corr, len(sent), iob_out, float(loss)


def analyse_output(sent):
  ncorr, ntotal, iob_out, loss = test_sent(model, sent, featurizer, criterion)
  fixed_iob = fix(iob_out)
  iob_ann = "".join([tok["misc"]["iob"] for tok in sent])
  for i, tok in enumerate(s):
    if not iob_ann[i] == fixed_iob[i]:
      is_correct = "✗"
    else:
      is_correct = ""
    print("i: {0:<4} form: {1:<30} gold: {2:<4} sys: {3:<4}   {4:<4}".format(i, tok["form"], iob_ann[i], fixed_iob[i], is_correct))

