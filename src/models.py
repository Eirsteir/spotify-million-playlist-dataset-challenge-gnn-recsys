import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj


class IGMC(torch.nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
            self, 
            num_features,
            num_relations=1, 
            latent_dim=[32, 32, 32, 32], 
            num_layers=4, 
            num_bases=2, 
            adj_dropout=0.2,
            force_undirected=False, 
            use_features=False, 
            n_side_features=0
        ):
        super().__init__()

        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(num_features, latent_dim[0], num_relations, num_bases))
        
        for i in range(0, num_layers - 1):
            self.convs.append(RGCNConv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))

        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.use_features = use_features

        if use_features:
            self.lin1 = Linear(2*sum(latent_dim) + n_side_features, 128)

        self.lin2 = Linear(128, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )

        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 0] == 1
        movies = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[movies]], 1)

        if self.use_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x[:, 0]