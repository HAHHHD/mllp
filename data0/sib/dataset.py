import numpy as np
import pandas as pd
import torch_geometric as pyg
import torch_geometric.utils

import utils
from utils import *
import torch, os
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import time


class UnipartiteData(Data):
    def __init__(self, x, y, edge_index, edge_weight=None, **kawrgs):
        super(UnipartiteData, self).__init__(x=x, y=y, edge_index=edge_index,
                                             edge_weight=edge_weight,
                                             **kawrgs)


def scaling(c, b_l, A, b_u, l, u):
    b_u[b_u > 1e308] = np.inf
    b_l[b_l < -1e308] = -np.inf
    u[u > 1e308] = np.inf
    l[l < -1e308] = -np.inf

    # row scaling by b_l b_u
    scale_l = np.abs(b_l)
    scale_l[(scale_l == np.inf) | (scale_l == 0)] = 1
    scale_u = np.abs(b_u)
    scale_u[(scale_u == np.inf) | (scale_u == 0)] = 1
    scale_row = np.maximum(scale_l, scale_u)
    A = sparse_mat_div_by_vec(A, scale_row, 'row')
    b_l /= scale_row
    b_u /= scale_row

    # col scaling by A and  l u
    if np.isin(
            np.unique(np.abs(l)),
            [0, np.inf]
    ).all() and np.isin(
        (np.unique(np.abs(u))),
        [0, np.inf]
    ).all():
        print('u only contains 0/inf, correct for small medium generated')
    else:
        print('u l contains other values, correct for pp_engine presolved_medium')

    scale_l = np.abs(l)
    scale_l[(scale_l == np.inf) | (scale_l == 0)] = 1
    scale_u = np.abs(u)
    scale_u[(scale_u == np.inf) | (scale_u == 0)] = 1
    scale_col2 = np.maximum(1. / scale_l, 1. / scale_u)
    # assert np.isclose(scale_col2, 1).all(), 'for small medium, this is true'

    scale_col = np.abs(A).max(0).toarray().flatten()
    scale_col[(scale_col == np.inf) | (scale_col == 0)] = 1

    scale_col = np.maximum(scale_col, scale_col2)

    A = sparse_mat_div_by_vec(A, scale_col, 'col')
    A = A.tocsr()
    l *= scale_col
    u *= scale_col
    c = c / scale_col

    # if c still have large value, further scale it
    scale_c = np.abs(c).max()
    if scale_c == 0.:
        print('warning: all c is zero, check lp problem. Is this feasibility prob? ')
        scale_c = 1.
    c /= scale_c

    return c, b_l, A, b_u, l, u


def cvt_to_features(c, b_l, A, b_u, l, u, ):
    nrows, ncols = A.shape
    v_features = concatenate_on_lst_dim(
        c,
        count_nonzero_sparse_mat(A, 'col') / nrows,
        cos_sim_vec_and_sparse_mat(b_l, A),
        cos_sim_vec_and_sparse_mat(b_u, A),
        expand_inf(l), expand_inf(u)
    )
    c_features = concatenate_on_lst_dim(
        cos_sim_sparse_mat_and_vec(A, c),
        count_nonzero_sparse_mat(A, 'row') / ncols,
        cos_sim_sparse_mat_and_vec(A, l),
        cos_sim_sparse_mat_and_vec(A, u),
        expand_inf(b_l), expand_inf(b_u),
    )

    return v_features, c_features


class LPDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, load_meta=False, chunk=None):
        self.root = root
        self.load_meta = load_meta
        self.chunk = chunk # actually not used 
        self.p = self.q = 8
        super().__init__(root, transform, pre_transform, pre_filter)

    def dump_size_info(self, dst): 
        # note: must be whole dataset and to bipartitie ds.dump_size_info
        df=self.cache_size_info() 
        if len(self.indices()) != df.shape[0]: 
            logging.warning('sub-dataset get info, please use full dataset ') 
        if osp.exists(dst): return  
        train_ds, val_ds = split_train_val(self, seed=0) # seed 0 
        df.loc[train_ds.indices(), 'split']='train'
        df.loc[val_ds.indices(), 'split']='val'  
        df_dump(df, dst)
        return df 

    def cache_size_info(self, recache=False):
        from torch_geometric.loader import DataLoader
        dump_fn = self.root + '/size.json'
        try:
            if recache: raise ValueError('will recache')
            res = json_load(dump_fn)
        except Exception as e:
            logging.info(f'err {e}, recahche')
            loader = DataLoader(self, batch_size=1, num_workers=8, shuffle=False,
                                )# get nproc
                                
            res = []
            for idx, data in tqdm(enumerate(loader)):
                # nedges = data.edge_index.shape[-1] // 2
                nedges = data.edge_index.nnz()
                nnodes = data.num_nodes
                fn = osp.basename(data.processed_path[0])
                ei = data.edge_index.clone()
                ei = ei.set_value(torch.ones(nedges), layout='coo')
                vars_deg = ei.sum(0)
                cons_deg = ei.sum(1)
                r = dict(
                    idx=idx, nedges=nedges, nnodes=nnodes,
                    fn=fn,
                    ncons=data.x_s.shape[0], nvars=data.x_t.shape[0],
                    density=data.edge_index.density(),
                    num_basis_vars=(data.y_t == 1).sum().item(),
                )
                for sta in []: # 'mean', 'std', 
                    r[f'vars_deg_{sta}'] = stat(vars_deg)[sta].item()
                    r[f'cons_deg_{sta}'] = stat(cons_deg)[sta].item()
                res.append(r)
            json_dump(res, dump_fn)
        df=pd.DataFrame(res).loc[list(self.indices()), :] 
        if len(self.indices()) != df.shape[0]: 
            logging.warning('sub-dataset get info, please use full dataset ')
        df['fn'] = df.fn.str.replace('.pk', '', regex=False)
        return df 

    @property
    def raw_file_names(self):
        fns = []
        for folder in [self.raw_dir, self.processed_dir]:
            if not osp.exists(folder): continue
            now = os.listdir(folder)
            now = filter(lambda x: x.endswith('.pk'), now)
            now = list(now)
            now = sorted(now, key=lambda nm: (len(nm), nm))
            if len(now) > len(fns): fns = now
        # fns=split_out_chunk(fns, self.chunk) # may not need, because possibly 1). no pk exists 2). all pk exists due to runed already
        # fns=[fns[0]] # todo 
        if len(fns)==0: raise ValueError('not found pk')
        return fns

    @property
    def processed_file_names(self):
        raws = self.raw_file_names
        return raws

    def process(self, ):
        for idx, raw_path in tqdm(enumerate(self.raw_paths)):  
            processed_path = osp.join(self.processed_dir,
                                      osp.basename(raw_path)
                                      )
            # skip exist 
            if osp.exists(processed_path) and osp.exists(processed_path+'.meta'): 
                continue 
            # print('proc ', time.time(), idx, raw_path)
            # Read data from `raw_path`.
            [c, b_l, (row, col, data), b_u, l, u,
             con_lbls, var_lbls, con_nms, var_nms] = msgpack_load(raw_path, copy=True)
            # print(np.unique(c))
            ncons, nvars = len(con_nms), len(var_nms)
            A = coo_matrix((data, (row, col)), shape=(ncons, nvars)).tocsr()
            c, b_l, A, b_u, l, u = scaling(c, b_l, A, b_u, l, u)
            ncons, nvars = A.shape
            v_feas, c_feas = cvt_to_features(c, b_l, A, b_u, l, u, )
            v_feas, c_feas = torch.FloatTensor(v_feas), torch.FloatTensor(c_feas)

            y_s = torch.LongTensor((con_lbls))
            y_t = torch.LongTensor((var_lbls))

            cons_l_mask, cons_u_mask = c_feas[:, -3].abs().bool(), c_feas[:, -1].abs().bool()
            vars_l_mask, vars_u_mask = v_feas[:, -3].abs().bool(), v_feas[:, -1].abs().bool()
            assert (y_s[cons_l_mask] != 0).all()
            assert (y_s[cons_u_mask] != 2).all()
            violates = (y_t[vars_l_mask]==0).sum() 
            if violates!=0: logging.warning(f'violate {violates}')                
            assert (y_t[vars_u_mask] != 2).all()

            A = A.tocoo()
            new_r, new_c, new_d = A.row, A.col, A.data

            
            msgpack_dump(
                [new_r, new_c, new_d, c_feas.numpy(), v_feas.numpy(),
                 y_s.numpy(), y_t.numpy(), ncons + nvars,
                 ], processed_path
            )
            msgpack_dump(
                dict(
                    num_cons=ncons, num_vars=nvars, raw_path=raw_path,
                    processed_path=processed_path,
                    con_nms=con_nms.tolist(), var_nms=var_nms.tolist(),
                ), processed_path + '.meta'
            )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        fn = osp.join(self.processed_dir, self.processed_file_names[idx])
        # print('load ', fn)
        [row, col, A_data, c_feas, v_feas,
         y_s, y_t, nnodes, ] = msgpack_load(fn, copy=True)  # False
        ncons = c_feas.shape[0]
        assert A_data.max() <= 1
        assert A_data.min() >= -1
        assert c_feas.max() <= 1
        assert c_feas.min() >= -1
        # assert v_feas.max() <= 1
        # assert v_feas.min() >= -1 # it is possible that this not hold for pp_engine
        (c_feas, v_feas, y_s, y_t) = tuple(map(torch.from_numpy,
                                               (c_feas, v_feas, y_s, y_t)))
        aux = dict(processed_path=fn)
        if self.load_meta:
            meta = msgpack_load(fn + '.meta', copy=False)
            aux.update(
                dict(con_nms=meta['con_nms'], var_nms=meta['var_nms'],
                     )
            )
        edge_attr = torch.from_numpy(A_data).float()
        edge_index = torch.from_numpy(np.asarray([row, col + ncons], dtype=np.int64))
        edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index, edge_attr)
        is_vars = torch.zeros((nnodes), dtype=torch.long)
        is_vars[ncons:] = 1
        data = UnipartiteData(
            x=torch.cat((c_feas, v_feas), dim=0),
            y=torch.cat((y_s, y_t), dim=0),
            is_vars=is_vars,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=nnodes, **aux
        )
        # convert to sparse at the end
        return data


from torch_geometric.transforms import BaseTransform
class MyToBipartite(BaseTransform):
    def __init__(self, dev='cpu', phase='train', thresh_num=np.inf):
        super().__init__()
        self.dev = dev
        self.phase = phase
        self.thresh_num = thresh_num

    def __call__(self, batch):
        if hasattr(batch, 'x_s'): # already is bipartite  
            return batch 
        # if the graph is larger than thresh_num, then do nothing 
        # if              smaller               , then transform to bipartite 
        if batch.edge_index.shape[-1] // 2 > self.thresh_num:
            return batch
        is_vars = batch.is_vars.bool()
        is_cons = ~is_vars
        nnodes = batch.num_nodes
        nvars = is_vars.sum().item()
        ncons = nnodes - nvars

        mapping_vec = torch.empty(ncons + nvars, dtype=torch.long)
        mapping_vec[is_cons] = torch.arange(ncons)
        mapping_vec[is_vars] = torch.arange(nvars) + ncons

        edge_index = batch.edge_index
        edge_index[0, :] = mapping_vec[edge_index[0, :]]
        edge_index[1, :] = mapping_vec[edge_index[1, :]]

        edge_mask = edge_index[0, :] < ncons
        assert edge_mask.sum().item() * 2 == edge_mask.shape[0]

        edge_index_l2r = edge_index[:, edge_mask].clone()
        edge_index_l2r[1, :] -= ncons
        l2r = SparseTensor.from_edge_index(edge_index_l2r,
                                           edge_attr=batch.edge_attr[edge_mask],
                                           sparse_sizes=(ncons, nvars),
                                           )
        del batch.edge_attr

        x_s = batch.x[is_cons, :]
        x_t = batch.x[is_vars, :]
        y_s = batch.y[is_cons]
        y_t = batch.y[is_vars]

        batch.edge_index = l2r  # .to(self.dev, non_blocking=True)
        batch.x_s = x_s  # .to(self.dev, non_blocking=True)
        batch.x_t = x_t  # .to(self.dev, non_blocking=True)
        batch.y_s = y_s  # .to(self.dev, non_blocking=True)
        batch.y_t = y_t  # .to(self.dev, non_blocking=True)
        del batch.x, batch.y

        if hasattr(batch, 'batch_size'):
            batch.bs = batch.batch_size
        else:
            batch.bs = batch.num_nodes
        t_batch_size = batch.is_vars[:batch.bs].sum().item()
        s_batch_size = batch.bs - t_batch_size
        batch.s_bs = s_batch_size
        batch.t_bs = t_batch_size

        del batch.is_vars

        if self.phase == 'train':
            del batch.batch
        return batch


if __name__ == '__main__':
    args = utils.Environment(
    )
    prefix = args.dataset_prefix
    shell(f'mkdir -p {args.dataset_processed_prefix}/processed/')
    ds = LPDataset(args.dataset_processed_prefix, load_meta=False,
                   transform=MyToBipartite(thresh_num=np.inf),
                   )
    if not args.skip_exist: ds.process() 
    if not args.skip_exist: ds.cache_size_info(recache=1) 

    train_ds, val_ds = split_train_val(ds, seed=args.seed)
    # loader_args = dict(batch_size=1,
    #                    num_workers=args.num_workers, shuffle=True,
    #                    drop_last=True, prefetch_factor=args.prefetch_factor
    #                    )
    # from torch_geometric.loader import DataLoader

    # train_loader = DataLoader(train_ds,
    #                           **loader_args
    #                           )
    
    # max_edges = 4e6 * 3
    # filter_large_graph(train_ds, max_num=max_edges)
    # filter_large_graph(val_ds, max_num=max_edges)
    # print(len(train_ds.indices()), len(val_ds.indices()), len(ds))
    # lrg_grph = 0
    # for idx_graphs, batched_graphs in enumerate(train_loader):
    #     print(batched_graphs)
    #     if hasattr(batched_graphs, 'x_s'):
    #         # print('procesd')
    #         pass
    #     else:
    #         # print('large gr')
    #         lrg_grph += 1
    #         print(lrg_grph)
    # print(lrg_grph)
