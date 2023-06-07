import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.hid_feats = hid_feats
        self.out_feats = out_feats

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, self.hid_feats // 8, 8, attn_drop=0.3)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 4, 4)
            for rel in rel_names}, aggregate='mean')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 4, 4)
            for rel in rel_names}, aggregate='mean')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.out_feats // 2, 2)
            for rel in rel_names}, aggregate='mean')

        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for i in range(3)])
        self.bns2 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for i in range(3)])


    def forward(self, blocks, inputs):
        h = self.conv1(blocks[0], inputs)
        self.rel_list = list(h.keys())
        h[self.rel_list[0]] = F.leaky_relu(self.bns[0](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.leaky_relu(self.bns2[0](h[self.rel_list[1]].view(-1, self.hid_feats)))

        h = self.conv2(blocks[1], h)
        h[self.rel_list[0]] = F.leaky_relu(self.bns[1](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.leaky_relu(self.bns2[1](h[self.rel_list[1]].view(-1, self.hid_feats)))

        h = self.conv3(blocks[2], h)
        h[self.rel_list[0]] = F.tanh(self.bns[2](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.tanh(self.bns2[2](h[self.rel_list[1]].view(-1, self.hid_feats)))

        h = self.conv4(blocks[3], h)
        h = {k: ((v.view(-1, self.out_feats))) for k, v in h.items()}

        return h

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, etypes):
        super().__init__()

        self.rgcn = RGCN(
            in_features, hidden_features, out_features, etypes)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.rgcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score