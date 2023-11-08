from copy import deepcopy
#from ortools.linear_solver import pywraplp
import sys, os, time, logging, glob
import os.path as osp
import random
import scipy
import numpy as np
import torch
import torch_geometric as pyg
from gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
import perturbations
import blackbox_diff
#from lap_solvers.lml import LML
from lml import LML
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data
from torch.nn import functional as F

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

def cosine_similarity(i, j):
    global norm
    global Qdot
    if norm[i] <= 1e-6 or norm[j] <= 1e-6:
        return 0
    #print(u@v / np.linalg.norm(u) / np.linalg.norm(v))
    #print(np.arccos(u@v / np.linalg.norm(u) / np.linalg.norm(v)) * 180 / 3.1415926)
    #if (np.arccos(u@v / np.linalg.norm(u) / np.linalg.norm(v)) * 180 / 3.1415926 < 85 or np.arccos(u@v / np.linalg.norm(u) / np.linalg.norm(v)) * 180 / 3.1415926 > 95):
        #print("----------------")
    numerator = QDot[i][j]
    denominator = norm[i] * norm[j]
    if abs(numerator) * 100 < denominator:
        return 0
    else:
        #print(numerator / denominator)
        return numerator / denominator

def msgpack_load(file, **kwargs):
    assert osp.exists(file)
    import msgpack, gc
    import msgpack_numpy as m
    if kwargs.pop('allow_np', True):
        kwargs.setdefault('object_hook', m.decode)
    kwargs.setdefault('use_list', True)
    kwargs.setdefault('raw', False)
    is_cp = kwargs.pop('copy', False)

    gc.disable()
    with open(file, 'rb') as f:
        try:
            res = msgpack.unpack(f, **kwargs)
        except  Exception  as e: 
            raise ValueError(f'{file} {e}')
    gc.enable()

    if is_cp:
        import copy
        res = copy.deepcopy(res)
    return res


def split_idxs_train_val(ngraphs, seed=0):
    ntrain_graphs = int(max(ngraphs * 7 / 10,1)) 
    np.random.seed(seed)
    idxs = np.random.permutation(ngraphs)
    train_idxs = np.sort(idxs[:ntrain_graphs])
    val_idxs = np.sort(idxs[ntrain_graphs:])
    return train_idxs, val_idxs


def split_train_val(ds, seed=0, ):    
    if seed!=0: 
        logging.warning('seed for train val not 0, will force set to 0')
        seed=0
    ngraphs = len(ds) 
    train_idxs, val_idxs = split_idxs_train_val(ngraphs, seed)
    logging.info(f'split into {len(train_idxs)} train {len(val_idxs)} val, seed: {seed}')
    return ds[train_idxs], ds[val_idxs] 


def get_netlib_dataloader_pack(normalize, device):
    data_list = []
    #-----------------------*********
    '''path = "processed/mps0000s"
    file_path = "dataset/mps0000s/"
    files = os.listdir(path)'''
    path = "../sib/lp-dataset/miplibCppp/highs-inp_tgt/raw"
    files = os.listdir(path)
    files = filter(lambda x: x.endswith('.pk'), files)
    files = list(files)
    files = sorted(files, key=lambda nm: (len(nm), nm))
    files = np.array(files)
    print(files)
    train_files, val_files = split_train_val(files, 0)
    files = train_files

    dataset = []
    train_dict = {}
    train_dict["obj"] = []
    train_dict["acc"] = []
    file_num = 0
    for file in files:
        if file == 'prob_2.mps':
            continue
        if file_num >= 10:
            break


        #---------------------------- for standard form ------------------------------------------
        #if normalize:
            #-----------------------*********
            #file_path = "dataset/MMIIXX/"
        #else:
            #file_path = "dataset/netlib_mps/"

        #v_basis_list = np.load(file_path+file+"_v.npy")
        #c_basis_list = np.load(file_path+file+"_c.npy")
        #basis_opt = np.concatenate([v_basis_list, c_basis_list])
        #basis_opt = np.load(file_path+file+"_basis.npy")

        '''basis_opt = np.load(file_path+file+"_v.npy")
        print(sum(basis_opt))
        coefs = np.load(file_path+file+"_coefs.npy")
        print(basis_opt.shape)
        if (coefs.shape[0] > 120000000):
            continue
        else:
            file_num += 1
        #print(file_num)
        #continue
        coefs = np.concatenate([coefs, np.array([0])])
        rhs = np.load(file_path+file+"_rhs.npy")
        rhs = np.expand_dims(rhs, axis=-1)
        #constrs_matrix = scipy.sparse.load_npz(file_path+file+"_constrs.npz").todense()
        constrs_matrix = scipy.sparse.load_npz(file_path+file+"_constrs.npz")

        constrs_matrix = scipy.sparse.coo_matrix(constrs_matrix)
        rhs = scipy.sparse.coo_matrix(rhs)
        assert(constrs_matrix.shape[0] == rhs.shape[0])
        assert(constrs_matrix.shape[1] == coefs.shape[0]-1)
        #constrs_matrix = np.concatenate([constrs_matrix, rhs], axis = -1)
        constrs_matrix = scipy.sparse.hstack([constrs_matrix, rhs], format="coo").T

        """constrs_matrix = cp.asarray(constrs_matrix.T)
        print("QR1 time:", time.time() - start)
        start = time.time()
        Q, _ = cp.linalg.qr(constrs_matrix)
        print("QR2 time:", time.time() - start)
        start = time.time()
        Q = cp.asnumpy(Q)"""
        """print("QR1 time:", time.time() - start)
        start = time.time()
        Q, _ = np.linalg.qr(constrs_matrix.T)"""
        """constrs_matrix = scipy.sparse.coo_matrix(constrs_matrix.T)
        print("QR1 time:", time.time() - start)
        start = time.time()
        Q, R, E, rank = sparseqr.qr( constrs_matrix, economy = True)
        print("QR2 time:", time.time() - start)
        start = time.time()
        Q = Q.todense()"""

        print("---------------------Instance {} name: {} size: {}".format(file_num, file, constrs_matrix.shape))
        start = time.time()
        #Q = constrs_matrix.tocsc() * 1e3
        constrs_matrix *= 1e3
        coefs *= 1e3
        if constrs_matrix.shape[0] > 82000:
            if constrs_matrix.shape[0] < 100000:
                constrs_matrix = constrs_matrix.tocsc()[:, :7000000000 // constrs_matrix.shape[0]].tocoo()
            elif constrs_matrix.shape[0] < 120000:
                constrs_matrix = constrs_matrix.tocsc()[:, :7500000000 // constrs_matrix.shape[0]].tocoo()
            elif constrs_matrix.shape[0] < 160000:
                constrs_matrix = constrs_matrix.tocsc()[:, :8000000000 // constrs_matrix.shape[0]].tocoo()
            else:
                constrs_matrix = constrs_matrix.tocsc()[:, :8500000000 // constrs_matrix.shape[0]].tocoo()
        #constrs_matrix = constrs_matrix.todense()
        print("QR00 time:", time.time() - start)
        start = time.time()
        #file_num += 1

        values = constrs_matrix.data
        indices = np.vstack((constrs_matrix.row, constrs_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = constrs_matrix.shape
        Q = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to("cuda")
        Q = Q.to_dense()
        torch.cuda.empty_cache()
        #norm = scipy.sparse.linalg.norm(Q, axis=1).reshape(-1, 1)
        norm = torch.linalg.norm(Q, axis=1)
        #print("QR02 time:", time.time() - start)
        #start = time.time()
        #Q = Q / norm
        #coefs = (coefs.reshape(-1, 1) / norm).squeeze(-1)


        #---------------------------
        tmp = (abs(norm) - 1e-6) < 0
        norm += tmp * 1e8
        #Q /= norm.unsqueeze(0).T
        #coefs = (np.expand_dims(coefs, axis=0).T / norm.unsqueeze(0).T.to('cpu').numpy()).T.squeeze()
        for i in range(coefs.shape[0]):
            if norm[i] > 1e-6:
                coefs[i] /= norm[i]
                Q[i] /= norm[i]
            #else:
                #print("111opps!!!!!!!!!!!!!!!!!!")
        if (np.linalg.norm(coefs) > 1e-6):
            coefs /= np.linalg.norm(coefs)
        #print(Q)
        print("QR1 time:", time.time() - start)
        start = time.time()
        del norm
        torch.cuda.empty_cache()
        #Q *= 1e3
        #norm = scipy.sparse.linalg.norm(Q, axis = 0).reshape(-1, 1)
        #Q = (Q.T / norm).T
        #del norm
        #torch.cuda.empty_cache()
        #print(constrs_matrix.T)
        Q, R = torch.linalg.qr(Q)
        #print(R)
        #print(Q)'''
        #---------------------------- for standard form ------------------------------------------


        #---------------------------- for general form ------------------------------------------
        start = time.time()
        [c, b_l, (row, col, data), b_u, l, u,
            con_lbls, var_lbls, con_nms, var_nms] = msgpack_load(path+'/'+file, copy=True)
        #print(c.dtype, b_l.dtype, data.dtype, b_u.dtype, l.dtype, u.dtype, con_lbls.dtype)
        c = c.astype(np.float32)
        b_l = b_l.astype(np.float32)
        b_u = b_u.astype(np.float32)
        l = l.astype(np.float32)
        u = u.astype(np.float32)
        data = data.astype(np.float32)
        #print(c.dtype, b_l.dtype, data.dtype, b_u.dtype, l.dtype, u.dtype, con_lbls.dtype)
        ncons, nvars = len(con_nms), len(var_nms)
        print("------------------------------instance:{}------------------------------".format(file_num + 1))
        print("size:", (nvars, ncons))
        mark1 = np.zeros(ncons + nvars + 1).astype(np.float32)
        mark2 = np.zeros(ncons + nvars + 1).astype(np.float32)
        mark3 = np.zeros(ncons + nvars + 1).astype(np.float32)
        mark3 = np.zeros(ncons + nvars + 1).astype(np.float32)
        '''mark1 = np.zeros(ncons + nvars).astype(np.float32)
        mark2 = np.zeros(ncons + nvars).astype(np.float32)
        mark3 = np.zeros(ncons + nvars).astype(np.float32)
        mark3 = np.zeros(ncons + nvars).astype(np.float32)
        mark4 = np.zeros(ncons + nvars).astype(np.float32)'''
        constrs_matrix = scipy.sparse.csc_matrix((data, (row, col)), shape=(ncons, nvars))
        row = np.array([i for i in range(ncons)])
        col = np.array([0 for i in range(ncons)])
        data = np.zeros(ncons)
        rhs = scipy.sparse.csc_matrix((data, (row, col)), shape=(ncons, 1))
        # concatenation
        row = np.array([i for i in range(ncons)])
        col = np.array([i for i in range(ncons)])
        data = np.ones(ncons)
        add = scipy.sparse.csc_matrix((data, (row, col)), shape=(ncons, ncons))
        constrs_matrix = scipy.sparse.hstack([constrs_matrix, add], format="csc")
        constrs_matrix = scipy.sparse.hstack([constrs_matrix, rhs], format="csc")
        c = np.array(list(c) + list(np.zeros(ncons + 1)))
        #c = np.array(list(c) + list(np.zeros(ncons)))
        # truncation
        if constrs_matrix.shape[1] > 82000:
            #continue
            if constrs_matrix.shape[1] < 100000:
                constrs_matrix = constrs_matrix[:7000000000 // constrs_matrix.shape[1], :]
            elif constrs_matrix.shape[1] < 120000:
                constrs_matrix = constrs_matrix[:7500000000 // constrs_matrix.shape[1], :]
            elif constrs_matrix.shape[1] < 160000:
                constrs_matrix = constrs_matrix[:8000000000 // constrs_matrix.shape[1], :]
            else:
                constrs_matrix = constrs_matrix[:8500000000 // constrs_matrix.shape[1], :]
        file_num += 1
        constrs_matrix = constrs_matrix.tocoo()
        values = constrs_matrix.data
        indices = np.vstack((constrs_matrix.row, constrs_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = constrs_matrix.shape
        constrs_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to("cuda")
        constrs_matrix = constrs_matrix.to_dense()
        torch.cuda.empty_cache()
        #print(constrs_matrix, constrs_matrix.shape)
        print("QR1 time:", time.time() - start)
        start = time.time()
        #print(c, b_l, constrs_matrix, b_u, l, u, con_lbls, var_lbls, rhs)
        
        vec = np.zeros(nvars)
        # scaling
        norm = torch.linalg.norm(constrs_matrix[:, :nvars], axis = 0)
        tmp = (abs(norm) - 1e-6) < 0
        norm += tmp * 1e8
        normy = norm.detach().cpu().numpy()
        c[:nvars] /= normy
        l *= normy
        u *= normy
        constrs_matrix[:, :nvars] /= norm

        # strange! This operation boosts the training acc
        for i in range(nvars):
            if l[i] != -np.inf and u[i] != np.inf:
                u[i] = -l[i]


        print("QR1 time:", time.time() - start)
        start = time.time()

        for i in range(nvars):
            # reflection
            if c[i] < 0 or (c[i] == 0 and torch.sum(constrs_matrix[:, i]) < 0):
                c[i] = -c[i]
                constrs_matrix[:, i] = - constrs_matrix[:, i]
                l[i], u[i] = -u[i], -l[i]
                var_lbls[i] = 2 - var_lbls[i]
            # shifting and mark
            if l[i] == -np.inf:
                if u[i] != np.inf:
                    #constrs_matrix[:, -1] = constrs_matrix[:, -1] - u[i] * constrs_matrix[:, i]
                    vec[i] = -u[i]
                    mark1[i] = -1
                    mark2[i] = 0
                    mark3[i] = 0
                else:
                    mark1[i] = -1
                    mark2[i] = 0
                    mark3[i] = 1
            elif u[i] != np.inf:
                #offset = (l[i] + u[i]) / 2
                #constrs_matrix[:, -1] = constrs_matrix[:, -1] - offset * constrs_matrix[:, i]
                vec[i] = -(l[i] + u[i]) / 2
                mark1[i] = 0
                mark2[i] = (-l[i] + u[i]) / 2
                mark3[i] = 0
            else:
                #constrs_matrix[:, -1] = constrs_matrix[:, -1] - l[i] * constrs_matrix[:, i]
                vec[i] = -l[i]
                mark1[i] = 0
                mark2[i] = 0
                mark3[i] = 1

        #print(constrs_matrix[:, -1].shape, constrs_matrix[:, :nvars].shape, vec.shape, torch.mm(constrs_matrix[:, :nvars], vec).shape)
        constrs_matrix[:, -1] += torch.mm(constrs_matrix[:, :nvars], torch.tensor(vec).to('cuda').float().unsqueeze(-1))[:, 0]


        print("QR1 time:", time.time() - start)
        start = time.time()

        # constraints
        for i in range(ncons):
            # reflection
            b_l[i], b_u[i] = -b_u[i], -b_l[i]
            con_lbls[i] = 2 - con_lbls[i]
            # shifting and mark
            if i < constrs_matrix.shape[0]:
                if b_l[i] == -np.inf:
                    if b_u[i] != np.inf:
                        constrs_matrix[:, -1][i] += -b_u[i]
                        b_u[i] = 0
                        mark1[nvars + i] = -1
                        mark2[nvars + i] = 0
                        mark3[nvars + i] = 0
                    else:
                        mark1[nvars + i] = -1
                        mark2[nvars + i] = 0
                        mark3[nvars + i] = 1
                elif b_u[i] != np.inf:
                    offset = -(b_l[i] + b_u[i]) / 2
                    constrs_matrix[:, -1][i] += offset
                    b_l[i] += offset
                    b_u[i] += offset
                    mark1[nvars + i] = 0
                    mark2[nvars + i] = b_u[i]
                    mark3[nvars + i] = 0
                else:
                    constrs_matrix[:, -1][i] += -b_l[i]
                    b_l[i] = 0
                    mark1[nvars + i] = 0
                    mark2[nvars + i] = 0
                    mark3[nvars + i] = 1
        # coef and constrs_matrix[:, -1]
        norm = np.linalg.norm(c)
        if norm > 1e-6:
            c = c / norm
        norm = torch.linalg.norm(constrs_matrix[:, -1], axis=0)
        if norm > 1e-6:
            constrs_matrix[:, -1] /= norm
        print("QR1 time:", time.time() - start)
        start = time.time()

        '''for i in range(nvars):
            if l[i] == -np.inf:
                if u[i] != np.inf:
                    mark1[i] = -1
                    mark2[i] = 0
                    mark3[i] = 0
                    mark4[i] = u[i]
                else:
                    mark1[i] = -1
                    mark2[i] = 0
                    mark3[i] = 1
                    mark4[i] = 0
            elif u[i] != np.inf:
                #offset = (l[i] + u[i]) / 2
                #constrs_matrix[:, -1] = constrs_matrix[:, -1] - offset * constrs_matrix[:, i]
                mark1[i] = 0
                mark2[i] = l[i]
                mark3[i] = 0
                mark4[i] = u[i]
            else:
                #constrs_matrix[:, -1] = constrs_matrix[:, -1] - l[i] * constrs_matrix[:, i]
                mark1[i] = 0
                mark2[i] = l[i]
                mark3[i] = 1
                mark4[i] = 0
        # coef and constrs_matrix[:, -1]
        norm = np.linalg.norm(c)
        if norm > 1e-6:
            c = c / norm
        print("QR1 time:", time.time() - start)
        start = time.time()'''

        #print(c, b_l, constrs_matrix, b_u, l, u, con_lbls, var_lbls, rhs)
        Q, R = torch.linalg.qr(constrs_matrix.T)
        #print(R)
        #print(Q)
        #---------------------------- for general form ------------------------------------------
        

        
        
        print("QR2 time:", time.time() - start)
        start = time.time()
        del R
        #Q = Q.to('cpu')
        torch.cuda.empty_cache()
        #Q = Q.cpu().numpy()
        '''tmp = abs(Q)
        idx = []
        torch.cuda.empty_cache()
        row = tmp.shape[0]
        num = 10
        for i in range(num + 1):
            if tmp.shape[0] == 0:
                continue
            tmp0 = tmp[:row // num] - 1e-5
            tmp0 = tmp0 > 0
            tmp = tmp[row // num:]
            torch.cuda.empty_cache()
            idx0 = torch.Tensor.nonzero(tmp0)
            idx0[:, 0] += row // num * i
            del tmp0
            torch.cuda.empty_cache()
            #print(idx0)
            if i:
                idx = torch.cat([idx, idx0], dim=0)
            else:
                idx = idx0
            del idx0
            torch.cuda.empty_cache()
        idx = idx.T
        #print(idx)
        del tmp
        torch.cuda.empty_cache()
        data = Q[idx[0],idx[1]]
        Q = torch.sparse_coo_tensor(idx, data, Q.shape)
        del idx
        del data
        torch.cuda.empty_cache()'''
        #print(Q)
        #row = Q._indices()[0].cpu()
        #col = Q._indices()[1].cpu()
        #data = Q._values().cpu()
        #shape = Q.size()
        #Q = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
        #print(Q)
        #Q = np.array(Q.cpu().numpy())
        print(Q.shape)
        #print(constrs_matrix)
        #print(Q, coefs)
        print("QR3 time:", time.time() - start)

        # graph construction
        is_var = np.ones(Q.shape[0])
        start = time.time()
        # split Q into num * num parts
        num = 16
        #tol_row = Q.shape[0] - 1
        '''indices = torch.from_numpy(np.vstack((Q.row, Q.col)).astype(np.int64))
        values = torch.from_numpy(Q.data)
        shape = torch.Size(Q.shape)
        Q = torch.sparse.FloatTensor(indices, values, shape)
        del indices
        del values
        del shape
        torch.cuda.empty_cache()
        print("Dot1 time:", time.time() - start)
        start = time.time()'''
        #Q = Q.to_dense().to(device)
        #torch.cuda.empty_cache()
        #start = time.time()
        tol_row = Q.shape[0]
        #Q = Q.to_dense()
        #norm = torch.linalg.norm(Q, axis=1).to(device)
        norm = []
        Qa = [[] for i in range(num)]
        na = [[] for i in range(num)]
        tmp = []
        thresh = 0.01    #0.05
        for i in range(num):
            partition = (i + 1) * tol_row // num - i * tol_row // num
            Qa[i] = Q[: partition]
            Q = Q[partition: ]
            torch.cuda.empty_cache()
            na[i] = torch.linalg.norm(Qa[i], axis=1)
            if len(norm):
                norm = np.hstack((norm, na[i].cpu().numpy()))
            else:
                norm = na[i].cpu().numpy()
            tmp = (abs(na[i]) - 1e-6) < 0
            na[i] += tmp * 1e8
            #Qa[i] = Qa[i] / na[i].unsqueeze(0).T
        del Q
        _edge_index = _edge_attr = 0
        print("Dot1 time:", time.time() - start)
        start = time.time()
        for i in range(num * num):
            row = i // num
            col = i % num
            if row > col:
                continue
            QQ1 = (Qa[row] / na[row].unsqueeze(0).T).to("cuda") #np.expand_dims(na[row], axis=0).T
            QQ2 = (Qa[col] / na[col].unsqueeze(0).T).to("cuda") #np.expand_dims(na[col], axis=0).T
            #QDot = torch.matmul(Qa[row].to("cuda"), Qa[col].T.to("cuda"))
            QDot = torch.matmul(QQ1, QQ2.T)
            del QQ1
            del QQ2
            torch.cuda.empty_cache()
            max_num = 30000
            if (row == col):
                QDot = torch.triu(QDot, diagonal=1)

            edge_index = torch.Tensor.nonzero(abs(QDot) > thresh)
            if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.4):
                thresh = 0.03
                edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.4):
                    thresh = 0.05
                    edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                    if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.4):
                        thresh = 0.1
                        edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                        if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.4):
                            thresh = 0.2
                            edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                            if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.4):
                                thresh = 0.3
                                edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                                if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.3):
                                    thresh = 0.4
                                    edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
                                    if (edge_index.shape[0] > max_num and max_num / edge_index.shape[0] < 0.3):
                                        thresh = 0.5
                                        edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))
            torch.cuda.empty_cache()
            edge_index = edge_index.T
            #print(thresh)
            #print("--------------------")
            if edge_index.shape[1] > max_num:
                print(thresh, max_num / edge_index.shape[1])
                edge_index = edge_index[:, : 3 * max_num]
                #print(edge_index.shape)
                edge_attr = abs(QDot[edge_index[0], edge_index[1]].unsqueeze(0))
                #print(edge_attr.shape)
                _, selected_index = edge_attr.topk(max_num, dim=1, largest=True)
                #print(selected_index.shape)
                edge_index = edge_index[:, selected_index[0]]
            #if edge_index.shape[0] > max_num:
                #print(max_num / edge_index.shape[0])
                #edge_index = torch.cat([edge_index[:max_num // 2], edge_index[edge_index.shape[0] - max_num // 2:]], dim=0)
            edge_attr = QDot[edge_index[0], edge_index[1]].unsqueeze(-1)
            del QDot
            torch.cuda.empty_cache()
            #print(edge_index)
            #print(edge_attr)
            if i == 0:
                _edge_index = edge_index
                _edge_attr = edge_attr
            else:
                edge_index[0] = edge_index[0] + row * tol_row // num
                edge_index[1] = edge_index[1] + col * tol_row // num
                #print(edge_index.shape, _edge_index.shape)
                #print(edge_index, _edge_index)
                #print(edge_attr, _edge_attr)
                _edge_index = torch.cat([_edge_index, edge_index], dim=1)
                _edge_attr = torch.cat([_edge_attr, edge_attr], dim=0)
            torch.cuda.empty_cache()
        print("Dot2 time:", time.time() - start)
        #print("edge counts:", edge_index.shape)
        #print("edge_attr:", edge_attr.shape)
        """cons_cos = np.dot(Q[:-1] / np.expand_dims(norm[:-1], axis=0).T, Q[-1].T).T
        tmp = (abs(cons_cos) - 0.01 * norm[-1]) > 0
        cons_cos = np.multiply(cons_cos, tmp) * 1e6
        if np.sum(cons_cos) < 1e-6:
            cons_cos = np.squeeze(np.zeros(cons_cos.shape[0]))
        else:
            cons_cos = np.squeeze(cons_cos * cons_cos.shape[0] / np.sum(cons_cos))
        print(coefs[:-1], norm[:-1], cons_cos)
        x = torch.tensor([coefs[:-1], norm[:-1], cons_cos], dtype=torch.float, device=device).T"""

        del Qa
        del na
        torch.cuda.empty_cache()
        
        '''is_var[-1] = 0
        x = torch.tensor(np.array([coefs, norm, is_var]), dtype=torch.float, device=device).T
        basis_num = torch.tensor(sum(basis_opt)).to(device)'''
        #-----------------------------------------------------------------
        is_var[nvars:nvars + ncons] = -1
        is_var[-1] = 0
        x = torch.tensor(np.array([c, norm, is_var, mark1, mark2, mark3]), dtype=torch.float, device=device).T
        basis_num = ncons
        basis_opt = np.concatenate((var_lbls, con_lbls))
        #basis_opt = np.concatenate((basis_opt, np.array([0])))

        #print(x[0].dtype, x[1].dtype, x[2].dtype, mark1.dtype, mark2.dtype, _edge_index.dtype, edge_attr.dtype, basis_opt.dtype)
        var_num = x.shape[0] - 1
        #var_num = x.shape[0]
        #data_list.append(pyg.data.Data(x=x, edge_index=_edge_index, edge_attr=_edge_attr, name=file, basis_opt=basis_opt, basis_num=basis_num, var_num=var_num))
        is_var = is_var.astype(int)
        data_list.append(Data(x=x, y=basis_opt, edge_index=_edge_index, edge_attr=_edge_attr, name=file, basis_num=basis_num, var_num=var_num, num_nodes=var_num+1, is_var=is_var))
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    return loader, train_dict



def get_netlib_dataloader(train_dataset, device):
    data_list = []
    i = 0
    for name, constr_Q, coefs, basis_opt in train_dataset:
        print("basis opt size", basis_opt.shape)
        i += 1
        print("-------------------instance:", i)
        data_list.append(build_graph_from_Q_sets(constr_Q, coefs, device, name, basis_opt))
    loader = DataLoader(data_list, batch_size=1)
    return loader

def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)


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

class AngleModel(torch.nn.Module):
    def __init__(self, feat_dim=16):
        super(AngleModel, self).__init__()
        self.gconv1 = pyg.nn.TransformerConv(6, feat_dim, edge_dim = 1) #------------
        self.gconv2 = pyg.nn.TransformerConv(feat_dim, feat_dim, edge_dim = 1)
        #self.gconv3 = pyg.nn.TransformerConv(feat_dim, feat_dim, edge_dim = 1)
        self.fc = torch.nn.Linear(feat_dim, 3) #------------
    
    def forward(self, g):
        x = torch.relu(self.gconv1(g.x, g.edge_index, g.edge_attr))
        x = torch.relu(self.gconv2(x, g.edge_index, g.edge_attr))
        #x = torch.relu(self.gconv3(x, g.edge_index, g.edge_attr))
        x = torch.relu(torch.nn.functional.dropout(x, p=.1, training=self.training))
        x = self.fc(x)
        #x = self.fc(x).squeeze()

        # knowledge-based mask
        #-----------------------------------------
        l_mask = (g.x[:, 3] == -1).bool()
        u_mask = (g.x[:, 5] == 1).bool()
        x = F.normalize(x) * 10
        x[l_mask, 0] = x[l_mask, 0] - 10
        x[u_mask, 2] = x[u_mask, 2] - 10
        #-----------------------------------------

        return x[:-1]#[:,1][:-1] #torch.sigmoid(x)

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
