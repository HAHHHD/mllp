import faulthandler
# 在import之后直接添加以下启用代码即可
print("ssss")
faulthandler.enable()
from linear_program_methods import *
import time
#import xlwt
from datetime import datetime
import torch_geometric as pyg
import os
import sys
import json
from config import load_config
from linear_program_data import get_random_dataset, get_twitch_dataset, get_facebook_dataset, get_netlib_dataset, get_netlib_dataset_dense
from sklearn.metrics import f1_score
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


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


path = "../sib/lp-dataset/mpsppp/highs-inp_tgt/raw"
highs_path = "highs/eval_mps_mps_bal/"
model_name = "model_#00422##.pth"
os.makedirs(highs_path, exist_ok = True)
files = os.listdir(path)
file_num = 0
files = list(files)
files = sorted(files, key=lambda nm: (len(nm), nm))
files = np.array(files)
print(files)
train_files, val_files = split_train_val(files, 0)
files = val_files

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type) 
set_seed()

skip = 0
obj_sum = 0
avg_f1 = 0
avg_acc = 0
avg_acc0 = 0
pre_time = []
QR_time = []
dot_time = []
inference_time = []
tol_time = time.time()
with torch.no_grad():
    for file in files:
        time_tmp = 0
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if file == 'prob_2.mps':
            continue
        if file_num >= 10000:
            break
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
        v_msk = np.zeros(nvars).astype(np.int64)
        con_lbls0, var_lbls0 = con_lbls.copy(), var_lbls.copy()
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
        start_time = time.time()
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

        for i in range(nvars):
            # reflection
            if c[i] < 0 or (c[i] == 0 and torch.sum(constrs_matrix[:, i]) < 0):
                c[i] = -c[i]
                constrs_matrix[:, i] = - constrs_matrix[:, i]
                l[i], u[i] = -u[i], -l[i]
                var_lbls[i] = 2 - var_lbls[i]
                v_msk[i] = 1
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

        # constraints
        b_l, b_u = -b_u, -b_l
        con_lbls = 2 - con_lbls
        for i in range(ncons):
            # reflection
            #b_l[i], b_u[i] = -b_u[i], -b_l[i]
            #con_lbls[i] = 2 - con_lbls[i]
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
        pre_time.append(time.time()-start_time)
        print("pre_time:", time.time() - start_time)

        torch.cuda.empty_cache()
        start_time = time.time()
        Q, R = torch.linalg.qr(constrs_matrix.T)
        QR_time.append(time.time()-start_time)
        print("QR_time:", time.time() - start_time)
        del constrs_matrix
        del R
        torch.cuda.empty_cache()
        Q = Q.to(torch.float16)
        
        start_time = time.time()
        is_var = np.ones(Q.shape[0])
        num = 16
        tol_row = Q.shape[0]
        norm = []
        Qa = [[] for i in range(num)]
        na = [[] for i in range(num)]
        tmp = []
        thresh = 0.01
        for i in range(num):
            slide = (i + 1) * tol_row // num - i * tol_row // num
            Qa[i] = Q[: slide]
            Q = Q[slide: ]
            torch.cuda.empty_cache()
            na[i] = torch.linalg.norm(Qa[i], axis=1)
            if len(norm):
                norm = np.hstack((norm, na[i].cpu().numpy()))
            else:
                norm = na[i].cpu().numpy()
            tmp = (abs(na[i]) - 1e-6) < 0
            na[i] += tmp * 1e8
        del Q
        _edge_index = _edge_attr = 0
        for i in range(num * num):
            row = i // num
            col = i % num
            if row > col:
                continue
            QQ1 = (Qa[row] / na[row].unsqueeze(0).T).to("cuda") #np.expand_dims(na[row], axis=0).T
            QQ2 = (Qa[col] / na[col].unsqueeze(0).T).to("cuda") #np.expand_dims(na[col], axis=0).T
            QDot = torch.matmul(QQ1, QQ2.T)
            del QQ1
            del QQ2
            torch.cuda.empty_cache()
            max_num = 30000   #72000
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
            if edge_index.shape[1] > max_num:
                edge_index = edge_index[:, : 3 * max_num]
                edge_attr = abs(QDot[edge_index[0], edge_index[1]].unsqueeze(0))
                _, selected_index = edge_attr.topk(max_num, dim=1, largest=True)
                edge_index = edge_index[:, selected_index[0]]
            edge_attr = QDot[edge_index[0], edge_index[1]].unsqueeze(-1)
            del QDot
            torch.cuda.empty_cache()
            if i == 0:
                _edge_index = edge_index
                _edge_attr = edge_attr
            else:
                edge_index[0] = edge_index[0] + row * tol_row // num
                edge_index[1] = edge_index[1] + col * tol_row // num
                _edge_index = torch.cat([_edge_index, edge_index], dim=1)
                _edge_attr = torch.cat([_edge_attr, edge_attr], dim=0)
        _edge_attr = _edge_attr.to(torch.float)
        dot_time.append(time.time()-start_time)
        print("Dot_time:",time.time()-start_time)
        del Qa
        del na
        torch.cuda.empty_cache()


        is_var[nvars:nvars + ncons] = -1
        is_var[-1] = 0
        x = torch.tensor(np.array([c, norm, is_var, mark1, mark2, mark3]), dtype=torch.float, device=device).T
        basis_num = ncons
        basis_opt = np.concatenate((var_lbls, con_lbls))
        #print(x[0].dtype, x[1].dtype, x[2].dtype, mark1.dtype, mark2.dtype, _edge_index.dtype, edge_attr.dtype, basis_opt.dtype)
        var_num = x.shape[0] - 1

        data_list = []
        data_list.append(pyg.data.Data(x=x, edge_index=_edge_index, edge_attr=_edge_attr, name=file, basis_opt=basis_opt, basis_num=basis_num, var_num=var_num))
        eval_loader = DataLoader(data_list, batch_size=1)
        del x
        del _edge_index
        del _edge_attr
        torch.cuda.empty_cache()
#------------------------------------------------------------------
        model = torch.load(model_name).to(device)
        model.eval()
        for _, graph in enumerate(eval_loader, 0):
            start_time = time.time()
            name, basis_num, var_num, basis_opt = graph.name[0], graph.basis_num[0], graph.var_num[0], torch.tensor(graph.basis_opt[0], dtype = torch.float, device = device)
            torch.cuda.empty_cache()
            latent_vars = model(graph)
            basis_opt = basis_opt.to(torch.long)
            if (sum(np.array(basis_opt.detach().cpu().numpy() == 3)) > 0):
                skip += 1
                continue
            pr = F.softmax(latent_vars.float(), dim=-1) 
            pr = pr.clone()
            pr[torch.isnan(pr)]=0 # if half model, nan will occur 
            _, topk_idx = pr[:, 1].topk(basis_num)
            pr[:, 1] = pr.min() - 1
            pr[topk_idx, 1] = pr.max() + 1
            pred = pr.argmax(-1)
            pred = pred.to(torch.long).cpu().detach().numpy()
            inference_time.append(time.time()-start_time)
            print("inf_time:",time.time()-start_time)
            basis_opt = basis_opt.cpu().detach().numpy()
            f1 = 0
            assert (np.sum(pred == 1) == np.sum(basis_opt == 1))
            acc = (pred == basis_opt).mean()
            avg_f1 += f1
            avg_acc += acc

            vbas = pred[: nvars]
            cbas = pred[nvars: ]
            assert np.sum(v_msk * (2 - var_lbls) + (1 - v_msk) * var_lbls != var_lbls0) == 0
            vbas = v_msk * (2 - vbas) + (1 - v_msk) * vbas
            cbas = 2 - cbas
            assert (np.sum(np.concatenate((vbas, cbas)) == 1) == np.sum(np.concatenate((var_lbls0, con_lbls0)) == 1))
            acc0 = (np.concatenate((vbas, cbas)) == np.concatenate((var_lbls0, con_lbls0))).mean()
            avg_acc0 += acc0
            print("eval result------------------------------------------------")
            print(f'%8d, %8d, %5f, %5f'%(basis_num, var_num, acc, acc0))
            f = open(highs_path+file+'.bas','w')
            f.write("HiGHS v1\nValid\n# Columns "+str(len(vbas))+"\n"+' '.join(map(str, vbas))+"\n# Rows "+str(len(cbas))+"\n"+' '.join(map(str, cbas)))
            f.close()
        del latent_vars
        del model
        torch.cuda.empty_cache()
        np.save(highs_path+"pre_"+file, np.array(pre_time))
        np.save(highs_path+"QR_"+file, np.array(QR_time))
        np.save(highs_path+"dot_"+file, np.array(dot_time))
        np.save(highs_path+"inf_"+file, np.array(inference_time))
        np.save(highs_path+"tol_"+file, np.array(QR_time) + np.array(dot_time) + np.array(inference_time))
        np.save(highs_path+"ttol_"+file, np.array(pre_time) + np.array(QR_time) + np.array(dot_time) + np.array(inference_time))
        print("instance done---------------")
print("final eval result-------------------------------------------")
print(f'len={file_num}-{skip}, avg_f1={avg_f1 / (file_num - skip)}, avg_acc={avg_acc / (file_num - skip)}, avg_acc0={avg_acc0 / (file_num - skip)},time={time.time() - tol_time}')
