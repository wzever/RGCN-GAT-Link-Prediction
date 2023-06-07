import torch
import dgl
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numpy.linalg import norm
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)

def read_txt(file):
    res_list = list()
    with open(file, "r") as f:
        line_list = f.readlines()
    for line in line_list:
        res_list.append(list(map(int, line.strip().split(' '))))      
    return res_list

def process_data():
    base_path = "./data/"

    cite_file = "paper_file_ann.txt"
    train_ref_file = "bipartite_train_ann.txt"
    test_ref_file = "bipartite_test_ann.txt"
    coauthor_file = "author_file_ann.txt"
    feature_file = "feature.pkl"

    citation = read_txt(os.path.join(base_path, cite_file))
    existing_refs = read_txt(os.path.join(base_path, train_ref_file))
    refs_to_pred = read_txt(os.path.join(base_path, test_ref_file))
    coauthor = read_txt(os.path.join(base_path, coauthor_file))

    feature_file = os.path.join(base_path, feature_file)
    with open(feature_file, 'rb') as f:
        paper_feature = pkl.load(f)

    cite_edges = pd.DataFrame(citation, columns=['source', 'target'])
    cite_edges = cite_edges.set_index(
        "c-" + cite_edges.index.astype(str)
    )

    ref_edges = pd.DataFrame(existing_refs, columns=['source', 'target'])
    ref_edges = ref_edges.set_index(
        "r-" + ref_edges.index.astype(str)
    )

    coauthor_edges = pd.DataFrame(coauthor, columns=['source', 'target'])
    coauthor_edges = coauthor_edges.set_index(
        "a-" + coauthor_edges.index.astype(str)
    )

    node_tmp = pd.concat([cite_edges.loc[:, 'source'], cite_edges.loc[:, 'target'], ref_edges.loc[:, 'target']])
    node_papers = pd.DataFrame(index=pd.unique(node_tmp))

    node_tmp = pd.concat([ref_edges['source'], coauthor_edges['source'], coauthor_edges['target']])
    node_authors = pd.DataFrame(index=pd.unique(node_tmp))

    train_refs = ref_edges.sample(frac=0.9,random_state=0,axis=0)
    test_true_refs = ref_edges[~ref_edges.index.isin(train_refs.index)]
    test_true_refs.loc[:, 'label'] = 1

    false_source = node_authors.sample(frac=test_true_refs.shape[0]/node_authors.shape[0],random_state=0,replace=True,axis=0)
    false_target = node_papers.sample(frac=test_true_refs.shape[0]/node_papers.shape[0],random_state=0,replace=True,axis=0)
    false_source = false_source.reset_index()
    false_target = false_target.reset_index()
    test_false_refs = pd.concat([false_source, false_target], axis=1)
    test_false_refs.columns = ['source', 'target']
    test_false_refs = test_false_refs[test_false_refs.isin(ref_edges) == False]
    test_false_refs.loc[:, 'label'] = 0

    test_refs = pd.concat([test_true_refs, test_false_refs.iloc[:min(len(false_source), len(false_target))]])
    test_refs = test_refs.sample(frac=1,random_state=0,axis=0)

    return train_refs, test_refs, refs_to_pred, cite_edges, coauthor_edges, paper_feature

def build_env(train_refs, cite_edges, coauthor_edges, paper_feature, device):
    os.environ['DGLBACKEND'] = 'pytorch'
    train_ref_tensor = torch.from_numpy(train_refs.values)
    cite_tensor = torch.from_numpy(cite_edges.values)
    coauthor_tensor = torch.from_numpy(coauthor_edges.values)
    rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), ('author', 'coauthor', 'author'), ('paper', 'beref', 'author')]
    graph_data = {
    rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
    rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
    rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
    rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0])
    }
    hetero_graph = dgl.heterograph(graph_data)

    author_feature = train_metapath(hetero_graph, ['ref', 'beref'], 'author').detach()

    node_features = {'author': author_feature, 'paper': paper_feature}

    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)

    return hetero_graph, rel_list

def train_metapath(g, path, ntype):
    print('Training metapath2vec as initial author feature...')
    model = dglnn.MetaPath2Vec(g, path, window_size=3, emb_dim=512)
    dataloader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=1024,
                        shuffle=True, collate_fn=model.sample)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.025)
    for epoch in tqdm(range(10)):
        for (pos_u, pos_v, neg_v) in dataloader:
            loss = model(pos_u, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    nids = torch.LongTensor(model.local_to_global_nid[ntype])
    emb = model.node_embed(nids)
    return emb

def compute_loss(pos_score, neg_score, etype):
    n_edges = pos_score[etype].shape[0]
    return (1 - pos_score[etype].unsqueeze(1) + neg_score[etype].view(n_edges, -1)).clamp(min=0).mean()

def Focal_loss(input, target, gamma=2):
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss
    
    return loss.sum(dim=1).mean()

def cos_sim(a, b):
    cos_sim = np.sum(a * b, axis = 1) / (norm(a, axis = 1) * norm(b, axis = 1))

    return cos_sim

def gen_csv_prediction(best_embed, refs_to_pred, threshold, test_f1, seed):
    # Output your prediction
    test_arr = np.array(refs_to_pred)
    res = cos_sim(np.array(best_embed['author'][test_arr[:, 0]]), np.array(best_embed['paper'][test_arr[:, 1]]))
    res[res >= threshold] = 1
    res[res < threshold] = 0
    data = []
    for index, p in enumerate(list(res)):
        tp = [index, str(int(p))]
        data.append(tp)

    if not os.path.exists('new_prediction'):
        os.makedirs('new_prediction')
    df = pd.DataFrame(data, columns=["Index", "Predicted"], dtype=object)
    df.to_csv(f'new_prediction/NEW_PREDICTION~{test_f1:.5f}-{seed}.csv', index=False)