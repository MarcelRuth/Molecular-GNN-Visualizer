import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch.nn import Linear
import torch

class GNNModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(feature_dim, hidden_dim)
        ] + [GCNConv(hidden_dim, hidden_dim) for _ in range(3)] + [
            GCNConv(hidden_dim, output_dim)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.convs[-1](x, edge_index)  # Last layer
        
        return x
    
class CBSModel(torch.nn.Module):
    def __init__(self, feature_dim,
                 hidden_dim):
        super(CBSModel, self).__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(feature_dim, hidden_dim)
        ] + [GCNConv(hidden_dim, hidden_dim) for _ in range(3)] + [
            GCNConv(hidden_dim, hidden_dim)
        ])

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.convs[-1](x, edge_index)  # Last layer

        x = torch.cat([global_add_pool(x, batch)], dim=1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
           
        return x

