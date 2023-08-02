from copy import deepcopy
#from ortools.linear_solver import pywraplp
import os
import random
import numpy as np
import torch
import torch_geometric as pyg
import time

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


####################################
#        helper functions          #
####################################

def compute_objective(weights, sets, selected_sets, bipartite_adj=None, device=torch.device('cpu')):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    if not isinstance(selected_sets, torch.Tensor):
        selected_sets = torch.tensor(selected_sets, device=device)

    if bipartite_adj is None:
        bipartite_adj = torch.zeros((len(sets), len(weights)), dtype=torch.float, device=device)
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1

    selected_items = bipartite_adj[selected_sets, :].sum(dim=-2).clamp_(0, 1)
    return torch.matmul(selected_items, weights)


def compute_obj_differentiable(weights, sets, latent_probs, bipartite_adj=None, device=torch.device('cpu')):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    if not isinstance(latent_probs, torch.Tensor):
        latent_probs = torch.tensor(latent_probs, device=device)
    if bipartite_adj is None:
        bipartite_adj = torch.zeros((len(sets), len(weights)), dtype=torch.float, device=device)
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1
    selected_items = torch.clamp_max(torch.matmul(latent_probs, bipartite_adj), 1)
    return torch.matmul(selected_items, weights), bipartite_adj


class BipartiteData(pyg.data.Data):
    def __init__(self, edge_index, x_src, x_dst, edge_attr):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x1 = x_src
        self.x2 = x_dst
        self.edge_attr = edge_attr

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x1.size(0)], [self.x2.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)

"""
def build_graph_from_weights_sets(weights, sets, device=torch.device('cpu')):
    x_src = torch.ones(len(sets), 1, device=device)
    x_tgt = torch.tensor(weights, dtype=torch.float, device=device).unsqueeze(-1)
    index_1, index_2 = [], []
    for set_idx, set_items in enumerate(sets):
        for set_item in set_items:
            index_1.append(set_idx)
            index_2.append(set_item)
    edge_index = torch.tensor([index_1, index_2], device=device)
    return BipartiteData(edge_index, x_src, x_tgt)
"""
def has_nan(x):
    return torch.sum(torch.isnan(x)) > 0

def build_graph_from_weights_sets(constrs, constr_weights, rhs, coefs, device=torch.device('cpu')):
    x_src = torch.tensor(coefs, dtype=torch.float, device=device).unsqueeze(-1)
    x_tgt = torch.tensor(rhs, dtype=torch.float, device=device).unsqueeze(-1)
    index_1, index_2, weights = [], [], []
    for constr_idx, vars in enumerate(constrs):
        for var_index in vars:
            index_1.append(var_index)
            index_2.append(constr_idx)
    
    for _, weight in enumerate(constr_weights):
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float, device=device).unsqueeze(-1)
    edge_index = torch.tensor([index_1, index_2], device=device)

    return BipartiteData(edge_index, x_src, x_tgt, weights)


#################################################
#         Learning Max-kVC Methods              #
#################################################

class InvariantModel(torch.nn.Module):
    def __init__(self, feat_dim, depth=2):
        super(InvariantModel, self).__init__()
        self.depth = depth
        self.feat_dim = feat_dim
        self.linear = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(feat_dim)) for _ in range(self.depth)])
        self.dir = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(feat_dim)) for _ in range(self.depth)])
        self.feat = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(feat_dim)) for _ in range(self.depth)])
        self.coef_embed = torch.nn.Linear(2, feat_dim)
        self.output = torch.nn.Linear(feat_dim, 1)
        self.reset_params()
        
    def reset_params(self):
        bd = 1 #/ self.feat_dim
        for i in range(self.depth):
            torch.nn.init.uniform_(self.linear[i], a=-bd, b=bd)
            torch.nn.init.uniform_(self.dir[i], a=-bd, b=bd)
            torch.nn.init.uniform_(self.feat[i], a=-bd, b=bd)
        
    def graph_block(self, X):
        X_update = torch.zeros(X.shape)
        for i in range(X.shape[0]):
            coef_i = X @ X[i]
            X_update[i] += X[i] + coef_i @ (X - X[i]) / X.shape[0]
        return X_update
    
    def forward(self, X, coefs):
        emb = X
        for i in range(self.depth):
            #emb = torch.einsum('i,jk->jik', self.linear[i], emb)
            """
            q = torch.einsum('i,jik->jk', self.feat[i], emb)
            k = torch.einsum('i,jik->jk', self.dir[i], emb)
            k_norm = k / torch.norm(k)
            inner_prod = torch.einsum('ik,ik->i',q, k_norm)
            sign = (inner_prod <= 0)
            k_scale = torch.einsum('i,i,ik->ik', inner_prod, sign, k_norm)
            emb = q - k_scale
            emb = self.graph_block(emb)
            """
        emb = torch.mean(emb, axis = 1)
        print(emb.shape)
        emb = torch.mean(emb @ emb.T, axis = -1)[:-1]
        return emb
        emb = torch.stack([emb, coefs], axis = -1)
        emb = self.coef_embed(emb)
        emb = self.output(emb)
        return emb.squeeze()

class GNNModel(torch.nn.Module):
    # Max covering model (3-layer Bipartite SageConv)
    def __init__(self):
        super(GNNModel, self).__init__()
        self.gconv1_w2s = pyg.nn.TransformerConv((1, 1), 16, edge_dim = 1)
        self.gconv1_s2w = pyg.nn.TransformerConv((1, 1), 16, edge_dim = 1)
        self.gconv2_w2s = pyg.nn.TransformerConv((16, 16), 16, edge_dim = 1)
        self.gconv2_s2w = pyg.nn.TransformerConv((16, 16), 16, edge_dim = 1)
        self.gconv3_w2s = pyg.nn.TransformerConv((16, 16), 16, edge_dim = 1)
        self.gconv3_s2w = pyg.nn.TransformerConv((16, 16), 16, edge_dim = 1)
        """
        self.gconv1_w2s = pyg.nn.SAGEConv((1, 1), 16)
        self.gconv1_s2w = pyg.nn.SAGEConv((1, 1), 16)
        self.gconv2_w2s = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv2_s2w = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv3_w2s = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv3_s2w = pyg.nn.SAGEConv((16, 16), 16)
        """
        self.fc = torch.nn.Linear(16, 1)
    """
    def forward(self, g):
        assert type(g) == BipartiteData
        reverse_edge_index = torch.stack((g.edge_index[1], g.edge_index[0]), dim=0)
        new_x1 = torch.relu(self.gconv1_w2s((g.x2, g.x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv1_s2w((g.x1, g.x2), g.edge_index))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv2_w2s((x2, x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv2_s2w((x1, x2), g.edge_index))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv3_w2s((x2, x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv3_s2w((x1, x2), g.edge_index))
        new_x = torch.cat([new_x1, new_x2], axis = 0)
        x = self.fc(new_x).squeeze()
        return x #torch.sigmoid(x)
    """

    def forward(self, g):
        assert type(g) == BipartiteData
        reverse_edge_index = torch.stack((g.edge_index[1], g.edge_index[0]), dim=0)
        new_x1 = torch.relu(self.gconv1_w2s((g.x2, g.x1), reverse_edge_index, g.edge_attr))#, g.edge_attr))#, edge_attr=g.edge_attr))
        new_x2 = torch.relu(self.gconv1_s2w((g.x1, g.x2), g.edge_index, g.edge_attr))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv2_w2s((x2, x1), reverse_edge_index, g.edge_attr))
        new_x2 = torch.relu(self.gconv2_s2w((x1, x2), g.edge_index, g.edge_attr))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv3_w2s((x2, x1), reverse_edge_index, g.edge_attr))
        #new_x2 = torch.relu(self.gconv3_s2w((x1, x2), g.edge_index, g.edge_attr))
        #new_x = torch.cat([new_x1, new_x2], axis = 0)
        x = self.fc(new_x1).squeeze()
        return x #torch.sigmoid(x)
