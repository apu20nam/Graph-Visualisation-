from dgl.data import FlickrDataset
import dgl
import torch


dataset = FlickrDataset()
graph = dataset[0]
nlabels = graph.ndata['label']
num_classes = dataset.num_classes

from gnnlens import Writer


num_nodes_to_include = 500
subgraph = dgl.node_subgraph(graph, torch.arange(num_nodes_to_include))


subgraph = dgl.add_self_loop(subgraph)

print(f"Subgraph has {subgraph.num_nodes()} nodes and {subgraph.num_edges()} edges.")

writer = Writer('tutorial_subgraph_nlabel')
writer.add_graph(name='Flickr_Subgraph', graph=subgraph, 
                 nlabels=subgraph.ndata['label'], 
                 num_nlabel_types=num_classes)

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# Define a class for GCN
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 num_classes,
                 num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, num_classes, allow_zero_in_degree=True))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(num_classes, num_classes, allow_zero_in_degree=True))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h

# Define a function to train a GCN with the specified number of layers 
# and return the predictions
def train_gcn(g, num_layers, num_classes):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    model = GCN(in_feats=features.shape[1],
                num_classes=num_classes,
                num_layers=num_layers)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
  
    num_epochs = 200
    model.train()
    for _ in range(num_epochs):
        logits = model(g, features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
    model.eval()
    predictions = model(g, features)
    _, predicted_classes = torch.max(predictions, dim=1)
    return predicted_classes

# Train and add predictions on the subgraph
print("Training GCN with one layer on subgraph...")
subgraph_predictions_one_layer = train_gcn(subgraph, num_layers=1, num_classes=num_classes)
print("Training GCN with two layers on subgraph...")
subgraph_predictions_two_layers = train_gcn(subgraph, num_layers=2, num_classes=num_classes)

writer.add_model(graph_name='Flickr_Subgraph', model_name='GCN_L1_Subgraph',
                 nlabels=subgraph_predictions_one_layer)
writer.add_model(graph_name='Flickr_Subgraph', model_name='GCN_L2_Subgraph',
                 nlabels=subgraph_predictions_two_layers)

# Finish dumping
writer.close()
