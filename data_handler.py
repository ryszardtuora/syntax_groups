import numpy
import conllu
import torch
from tqdm import tqdm

from treelstm import calculate_evaluation_orders

IOB = "IOB"
EMB_DIM = 300

POS_TAGS = ['adj', 'adja', 'adjc', 'adjp', 'adv', 'aglt', 'bedzie', 'brev', 'comp', 'conj', 'depr', 'dig', 'emo', 'fin', 'frag', 'ger', 'ign', 'imps', 'impt', 'inf', 'interj', 'interp', 'num', 'pact', 'pant', 'part', 'pcon', 'ppas', 'ppron12', 'ppron3', 'praet', 'pred', 'prep', 'romandig', 'siebie', 'subst', 'winien']

SIMPLE_DEPS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'expl', 'fixed', 'flat', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'root', 'vocative', 'xcomp']


def load_conllu(filepath):
    sentences = []
    print("loading sentences from {}".format(filepath))
    with open(filepath) as f:
        for s in tqdm(conllu.parse_incr(f)):
            if len(s) > 1:
                sentences.append(s)
    return sentences


def read_iob(sent, device):
    iob_ann = [IOB.index(tok["misc"]["iob"]) for tok in sent]
    return torch.tensor(iob_ann, dtype=torch.long, device=device)


def read_pos_tags(sent, device):
    tags = [token["xpostag"].split(":")[0] for token in sent]
    indices = [POS_TAGS.index(tag) for tag in tags]
    arrays = [numpy.zeros(37,) for i in indices]
    for i, a in zip(indices, arrays):
        a[i] = 1
    return arrays


def read_words(sent, emb, device):
    forms = [tok["form"] for tok in sent]
    x_list = []
    for form in forms:
        try:
            narray = emb[form]
        except KeyError:
            narray = numpy.zeros(EMB_DIM,)
        x_list.append(narray)
    return x_list


def read_deprels(sent, device):
    deprels = [tok["deprel"].split(":")[0] for tok in sent] # using simple deprels
    indices = [SIMPLE_DEPS.index(dep) for dep in deprels]
    arrays = [numpy.zeros(32,) for i in indices]
    for i, a in zip(indices, arrays):
        a[i] = 1
    return arrays
        


def read_adjacencies(sent, device):
    adjacencies = []
    for tok in sent:
        id = tok["id"] -1
        head = tok["head"] -1
        if head != -1:
            adjacencies.append((head, id))
    srt = sorted(adjacencies)
    return torch.tensor(adjacencies, dtype=torch.long, device=device)


def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))
    return adjacency_list


def convert_tree_to_tensors(tree, device):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)
    adjacency_list = sorted(adjacency_list)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }


def node_to_treenode(node, features, labels):
    nodedic = {}
    id = node.token["id"] -1
    nodedic["features"] = features[id]
    nodedic["labels"] = labels[id].item()
    nodedic["children"] = [node_to_treenode(c, features, labels) for c in node.children]
    return nodedic

def sent_to_tree(sent, features, labels, device):
    #try:
    tree = sent.to_tree()
    tree_dic = node_to_treenode(tree, features, labels)
    #try:
    converted = convert_tree_to_tensors(tree_dic, device)
    #except:
    #    print(sent)
    #    return {}
    return converted

def node_to_treenode(node, features, labels):
    nodedic = {}
    id = node.token["id"] -1
    nodedic["features"] = features[id]
    nodedic["labels"] = labels[id].item()
    nodedic["children"] = [node_to_treenode(c, features, labels) for c in node.children]
    return nodedic




