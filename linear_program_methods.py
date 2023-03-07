from copy import deepcopy
#from ortools.linear_solver import pywraplp
import os
import random
import numpy as np
import torch
import torch_geometric as pyg
import time
from gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
import perturbations
import blackbox_diff
from lap_solvers.lml import LML

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
        self.linear = []
        self.dir = []
        self.feat = []
        for _ in range(self.depth):
            self.linear.append(torch.nn.Parameter(torch.empty(feat_dim)))
            self.dir.append(torch.nn.Parameter(torch.empty(feat_dim)))
            self.feat.append(torch.nn.Parameter(torch.empty(feat_dim)))
        self.coef_embed = torch.nn.Linear(2, feat_dim)
        self.output = torch.nn.Linear(feat_dim, 1)
        self.reset_params()
        
    def reset_params(self):
        bd = 1 / self.feat_dim
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
            emb = torch.einsum('i,jk->jik', self.linear[i], emb)
            q = torch.einsum('i,jik->jk', self.feat[i], emb)
            k = torch.einsum('i,jik->jk', self.dir[i], emb)
            k_norm = k / torch.norm(k)
            inner_prod = torch.einsum('ik,ik->i',q, k_norm)
            sign = (inner_prod <= 0)
            k_scale = torch.einsum('i,i,ik->ik', inner_prod, sign, k_norm)
            emb = q - k_scale
            emb = self.graph_block(emb)
        print(emb)
        emb = torch.mean(emb @ emb.T, axis = -1)[:-1]
        print(emb)
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

def egn_max_covering(weights, sets, max_covering_items, model, egn_beta, random_trials=0, time_limit=-1):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, weights.device)
    graph.ori_x1 = graph.x1.clone()
    graph.ori_x2 = graph.x2.clone()
    best_objective = 0
    best_top_k_indices = None
    bipartite_adj = None
    for _ in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0:
            graph.x1 = graph.ori_x1 + torch.randn_like(graph.ori_x1) / 100
            graph.x2 = graph.ori_x2 + torch.randn_like(graph.ori_x2) / 100
        probs = model(graph).detach()
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        for prob_idx in probs_argsort:
            if selected_items >= max_covering_items:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            constraint_conflict_0 = torch.relu(probs_0.sum() - max_covering_items)
            constraint_conflict_1 = torch.relu(probs_1.sum() - max_covering_items)
            obj_0, bipartite_adj = compute_obj_differentiable(weights, sets, probs_0, bipartite_adj, device=probs.device)
            obj_0 = obj_0 - egn_beta * constraint_conflict_0
            obj_1, bipartite_adj = compute_obj_differentiable(weights, sets, probs_1, bipartite_adj, device=probs.device)
            obj_1 = obj_1 - egn_beta * constraint_conflict_1
            if obj_0 <= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = probs.nonzero().squeeze()
        objective = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=probs.device).item()
        if objective > best_objective:
            best_objective = objective
            best_top_k_indices = top_k_indices
    return best_objective, best_top_k_indices, time.time() - prev_time


def sinkhorn_max_covering(weights, sets, max_covering_items, model, sample_num, noise, tau, sk_iters, opt_iters, sample_num2=None, noise2=None, verbose=True):
    graph = build_graph_from_weights_sets(weights, sets, weights.device)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1
    if type(noise) == list and type(tau) == list and type(sk_iters) == list and type(opt_iters) == list:
        iterable = zip(noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([noise], [tau], [sk_iters], [opt_iters])
    for noise, tau, sk_iters, opt_iters in iterable:
        for train_idx in range(opt_iters):
            gumbel_weights_float = torch.sigmoid(latent_vars)
            # noise = 1 - 0.75 * train_idx / 1000
            top_k_indices, probs = gumbel_sinkhorn_topk(gumbel_weights_float, max_covering_items,
                max_iter=sk_iters, tau=tau, sample_num=sample_num, noise_fact=noise, return_prob=True)
            obj, bipartite_adj = compute_obj_differentiable(weights, sets, probs, bipartite_adj, probs.device)
            (-obj).mean().backward()
            if train_idx % 10 == 0 and verbose:
                print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(gumbel_weights_float, max_covering_items,
                max_iter=sk_iters, tau=tau, sample_num=sample_num2, noise_fact=noise2, return_prob=True)
            obj = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=probs.device)
            best_idx = torch.argmax(obj)
            max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            if max_obj > best_obj:
                best_obj = max_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = train_idx
            if train_idx % 10 == 0 and verbose:
                print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
            optimizer.step()
            optimizer.zero_grad()
    return best_obj, best_top_k_indices


def lml_max_covering(weights, sets, max_covering_items, model, opt_iters, verbose=True):
    graph = build_graph_from_weights_sets(weights, sets, weights.device)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    for train_idx in range(opt_iters):
        weights_float = torch.sigmoid(latent_vars)
        probs = LML(N=max_covering_items)(weights_float)
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj, bipartite_adj = compute_obj_differentiable(weights, sets, probs, bipartite_adj, probs.device)
        (-obj).mean().backward()
        if train_idx % 10 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        obj = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=probs.device)
        if obj > best_obj:
            best_obj = obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 10 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


def gumbel_max_covering(weights, sets, max_covering_items, model, sample_num, noise, opt_iters, verbose=True):
    graph = build_graph_from_weights_sets(weights, sets, weights.device)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=.001)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    @perturbations.perturbed(num_samples=sample_num, noise='gumbel', sigma=noise, batched=False, device=weights.device)
    def perturb_topk(covered_weights):
        probs = torch.zeros_like(covered_weights)
        probs[
            torch.arange(sample_num).repeat_interleave(max_covering_items),
            torch.topk(covered_weights, max_covering_items, dim=-1).indices.view(-1)
        ] = 1
        return probs
    for train_idx in range(opt_iters):
        gumbel_weights_float = torch.sigmoid(latent_vars)
        # noise = 1 - 0.75 * train_idx / 1000
        probs = perturb_topk(gumbel_weights_float)
        obj, bipartite_adj = compute_obj_differentiable(weights, sets, probs, bipartite_adj, probs.device)
        (-obj).mean().backward()
        if train_idx % 10 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=probs.device)
        best_idx = torch.argmax(obj)
        max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
        if max_obj > best_obj:
            best_obj = max_obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 10 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


def blackbox_max_covering(weights, sets, max_covering_items, model, lambda_param, opt_iters, verbose=True):
    graph = build_graph_from_weights_sets(weights, sets, weights.device)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    bb_topk = blackbox_diff.BBTopK()
    for train_idx in range(opt_iters):
        weights_float = torch.sigmoid(latent_vars)
        # noise = 1 - 0.75 * train_idx / 1000
        probs = bb_topk.apply(weights_float, max_covering_items, lambda_param)
        obj, bipartite_adj = compute_obj_differentiable(weights, sets, probs, bipartite_adj, probs.device)
        (-obj).mean().backward()
        if train_idx % 100 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=probs.device)
        if obj > best_obj:
            best_obj = obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 100 == 0 and verbose:
            print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


#################################################
#        Traditional Max-kVC Methods            #
#################################################

def greedy_max_covering(weights, sets, max_selected):
    sets = deepcopy(sets)
    covered_items = set()
    selected_sets = []
    for i in range(max_selected):
        max_weight_index = -1
        max_weight = 0

        # compute the covered weights for each set
        covered_weights = []
        for current_set_index, cur_set in enumerate(sets):
            current_weight = 0
            for item in cur_set:
                if item not in covered_items:
                    current_weight += weights[item]
            covered_weights.append(current_weight)
            if current_weight > max_weight:
                max_weight = current_weight
                max_weight_index = current_set_index

        assert max_weight_index != -1
        assert max_weight > 0

        # update the coverage status
        covered_items.update(sets[max_weight_index])
        sets[max_weight_index] = []
        selected_sets.append(max_weight_index)

    objective_score = sum([weights[item] for item in covered_items])

    return objective_score, selected_sets


def ortools_max_covering(weights, sets, max_selected, solver_name=None, linear_relaxation=True, timeout_sec=60):
    # define solver instance
    if solver_name is None:
        if linear_relaxation:
            solver = pywraplp.Solver('DAG_scheduling',
                                    pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        else:
            solver = pywraplp.Solver('DAG_scheduling',
                                    pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    else:
        solver = pywraplp.Solver.CreateSolver(solver_name)

    # Initialize variables
    VarX = {}
    VarY = {}
    ConstY = {}
    for item_id, weight in enumerate(weights):
        if linear_relaxation:
            VarY[item_id] = solver.NumVar(0.0, 1.0, f'y_{item_id}')
        else:
            VarY[item_id] = solver.BoolVar(f'y_{item_id}')

    for item_id in range(len(weights)):
        ConstY[item_id] = 0

    for set_id, set_items in enumerate(sets):
        if linear_relaxation:
            VarX[set_id] = solver.NumVar(0.0, 1.0, f'x_{set_id}')
        else:
            VarX[set_id] = solver.BoolVar(f'x_{set_id}')

        # add constraint to Y
        for item_id in set_items:
            #if item_id not in ConstY:
            #    ConstY[item_id] = VarX[set_id]
            #else:
                ConstY[item_id] += VarX[set_id]

    for item_id in range(len(weights)):
        solver.Add(VarY[item_id] <= ConstY[item_id])

    # add constraint to X
    X_constraint = 0
    for set_id in range(len(sets)):
        X_constraint += VarX[set_id]
    solver.Add(X_constraint <= max_selected)

    # the objective
    Covered = 0
    for item_id in range(len(weights)):
        Covered += VarY[item_id] * weights[item_id]

    solver.Maximize(Covered)

    if timeout_sec > 0:
        solver.set_time_limit(int(timeout_sec * 1000))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(len(sets))]
    else:
        print('Did not find the optimal solution. status={}.'.format(status))
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(len(sets))]


def gurobi_max_covering(weights, sets, max_selected, linear_relaxation=True, timeout_sec=60, start=None, verbose=True):
    import gurobipy as grb

    try:
        if type(weights) is torch.Tensor:
            tensor_input = True
            device = weights.device
            weights = weights.cpu().numpy()
        else:
            tensor_input = False
        if start is not None and type(start) is torch.Tensor:
            start = start.cpu().numpy()

        model = grb.Model('max covering')
        if verbose:
            model.setParam('LogToConsole', 1)
        else:
            model.setParam('LogToConsole', 0)
        if timeout_sec > 0:
            model.setParam('TimeLimit', timeout_sec)

        # Initialize variables
        VarX = {}
        VarY = {}
        ConstY = {}
        for item_id, weight in enumerate(weights):
            if linear_relaxation:
                VarY[item_id] = model.addVar(0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f'y_{item_id}')
            else:
                VarY[item_id] = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{item_id}')
        for item_id in range(len(weights)):
            ConstY[item_id] = 0
        for set_id, set_items in enumerate(sets):
            if linear_relaxation:
                VarX[set_id] = model.addVar(0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f'x_{set_id}')
            else:
                VarX[set_id] = model.addVar(vtype=grb.GRB.BINARY, name=f'x_{set_id}')
            if start is not None:
                VarX[set_id].start = start[set_id]

            # add constraint to Y
            for item_id in set_items:
                ConstY[item_id] += VarX[set_id]
        for item_id in range(len(weights)):
            model.addConstr(VarY[item_id] <= ConstY[item_id])

        # add constraint to X
        X_constraint = 0
        for set_id in range(len(sets)):
            X_constraint += VarX[set_id]
        model.addConstr(X_constraint <= max_selected)

        # the objective
        Covered = 0
        for item_id in range(len(weights)):
            Covered += VarY[item_id] * weights[item_id]
        model.setObjective(Covered, grb.GRB.MAXIMIZE)

        model.optimize()

        res = [model.getVarByName(f'x_{set_id}').X for set_id in range(len(sets))]
        if tensor_input:
            res = np.array(res, dtype=np.int)
            return model.getObjective().getValue(), torch.from_numpy(res).to(device)
        else:
            return model.getObjective().getValue(), res

    except grb.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
