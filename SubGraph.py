import dgl
from dgl.data import FlickrDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnnlens import Writer
from captum.attr import IntegratedGradients
from dgl.nn import GraphConv
from functools import partial

dataset = FlickrDataset()
graph = dataset[0]
nlabels = graph.ndata['label']
num_classes = dataset.num_classes

writer = Writer('tutorial_subgraph_2')

# Sample 2000 nodes because of large size of graph
sampled_nodes = torch.randperm(graph.num_nodes())[:2000]
subgraph = dgl.node_subgraph(graph, sampled_nodes)

# Add self-loops to handle zero in-degree nodes
subgraph = dgl.add_self_loop(subgraph)

# Update labels and other properties for the subgraph
subgraph_nlabels = subgraph.ndata['label']
subgraph_num_classes = len(torch.unique(subgraph_nlabels))


writer.add_graph(name='Flickr_Subgraph', graph=subgraph,
                 nlabels=subgraph_nlabels, num_nlabel_types=subgraph_num_classes)


class GCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, h, g):
        return self.conv(g, h)

# Required by IntegratedGradients
h = subgraph.ndata['feat'].clone().requires_grad_(True)
model = GCN(h.shape[1], subgraph_num_classes)
ig = IntegratedGradients(partial(model.forward, g=subgraph))

# Attribute the predictions for node class 0 to the input features
feat_attr = ig.attribute(h, target=0, internal_batch_size=subgraph.num_nodes(), n_steps=50)

# Compute node weights
node_weights = feat_attr.abs().sum(dim=1)
node_weights = (node_weights - node_weights.min()) / node_weights.max()

# Assign weights to subgraph nodes and edges
subgraph.ndata['weight'] = node_weights
subgraph.edata['weight'] = torch.randn(subgraph.num_edges(),)

# Define a subgraph extraction function
def extract_subgraph(g, node):
    seed_nodes = [node]
    sg = dgl.in_subgraph(g, seed_nodes)
    src, dst = sg.edges()
    seed_nodes = torch.cat([src, dst]).unique()
    sg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
    return sg

# Generate two subgraphs for visualization
first_subgraph = extract_subgraph(subgraph, 0)
writer.add_subgraph(graph_name='Flickr_Subgraph', subgraph_name='IntegratedGradients', node_id=0, 
                    subgraph_nids=first_subgraph.ndata[dgl.NID],
                    subgraph_eids=first_subgraph.edata[dgl.EID],
                    subgraph_nweights=first_subgraph.ndata['weight'],
                    subgraph_eweights=first_subgraph.edata['weight'])

second_subgraph = extract_subgraph(subgraph, 1)
writer.add_subgraph(graph_name='Flickr_Subgraph', subgraph_name='IntegratedGradients', node_id=1,
                    subgraph_nids=second_subgraph.ndata[dgl.NID],
                    subgraph_eids=second_subgraph.edata[dgl.EID],
                    subgraph_nweights=second_subgraph.ndata['weight'],
                    subgraph_eweights=second_subgraph.edata['weight'])


writer.close()
