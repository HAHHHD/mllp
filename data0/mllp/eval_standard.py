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


path = "processed/mpss"
file_path = "dataset/mpss/" #"dataset/netlib_mps_norm/"
highs_path = "highs/mpss_miplibCs/"
model_name = "model_#001.pth"
os.makedirs(highs_path, exist_ok = True)
files = os.listdir(path)
file_num = 0
files = list(files)
files = sorted(files, key=lambda nm: (len(nm), nm))
files = np.array(files)
print(files)
train_files, val_files = split_train_val(files, 0)
files = val_files

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type) 
set_seed()


obj_sum = 0
avg_f1 = 0
avg_acc = 0
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
        basis_opt = np.load(file_path+file+"_v.npy")
        print(sum(basis_opt))
        coefs = np.load(file_path+file+"_coefs.npy")
        print(basis_opt.shape)
        if (coefs.shape[0] > 4000000):
            continue
        else:
            file_num += 1
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
        constrs_num = constrs_matrix.shape[1]

        if constrs_matrix.shape[0] < 30000000:
            print("---------------------Instance {} name: {} size: {}".format(file_num, file, constrs_matrix.shape))
            start = time.time()
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
            values = constrs_matrix.data
            indices = np.vstack((constrs_matrix.row, constrs_matrix.col))
            i = torch.LongTensor(indices)
            v = torch.tensor(values, dtype=torch.float)
            shape = constrs_matrix.shape
            #Q0 = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to("cuda")
            #Q = Q0.to_dense()
            Q = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to("cuda")
            Q = Q.to_dense()
            print(Q.dtype)
            del i
            del v
            torch.cuda.empty_cache()
            start_time = time.time()
            norm = torch.linalg.norm(Q, axis=1)
            tmp = (abs(norm) - 1e-6) < 0
            norm += tmp * 1e8
            Q /= norm.unsqueeze(0).T
            coefs = (np.expand_dims(coefs, axis=0).T / norm.unsqueeze(0).T.to('cpu').numpy()).T.squeeze()
            if (np.linalg.norm(coefs) > 1e-6):
                coefs /= np.linalg.norm(coefs)
            print("QR1 time:", time.time() - start)
            start = time.time()
            del norm
            torch.cuda.empty_cache()
            Q, _ = torch.linalg.qr(Q)
            QR_time.append(time.time()-start_time)
            print("QR2 time:", time.time() - start)
            Q = Q.to(torch.float16)
            print(Q.dtype)
            print("QR2 time:", time.time() - start)
            start = time.time()
            del _
            torch.cuda.empty_cache()

        print("QR3 time:", time.time() - start)

        is_var = np.ones(Q.shape[0])
        start_time = time.time()
        start = time.time()
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
        print("Dot1 time:", time.time() - start)
        start = time.time()
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
        print("Dot2 time:", time.time() - start)
        del Qa
        del na
        torch.cuda.empty_cache()
#----------no QR-----------------------
        '''QDot = torch.sparse.mm(Q0, Q0.T)
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
                                    edge_index = torch.argwhere((QDot < -thresh) | (QDot > thresh))'''
        '''torch.cuda.empty_cache()
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
        print("Dot2 time:", time.time() - start)
        del Qa
        del na
        torch.cuda.empty_cache()'''


#-----------------------------------------------------------------




        is_var[-1] = 0
        x = torch.tensor([coefs, norm, is_var], dtype=torch.float, device=device).T
        print(x[0].dtype, x[1].dtype, x[2].dtype, _edge_index.dtype, edge_attr.dtype)
        var_num = x.shape[0] - 1
        basis_num = torch.tensor(sum(basis_opt)).to(device)
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
            latent_vars = model(graph)
            pred_indices = torch.topk(latent_vars, k=basis_num)[-1].cpu().detach().numpy()
            pred = np.zeros(var_num)
            pred[pred_indices] = 1
            f1 = f1_score(basis_opt.cpu().detach().numpy(), pred)
            correct_num = pred @ basis_opt.cpu().detach().numpy()
            acc = correct_num / basis_num
            avg_f1 += f1
            avg_acc += acc
            print("eval result------------------------------------------------")
            print(f'%8d, %8d, %8d, %5f'%(correct_num, basis_num, var_num, f1))
            pred_indices = torch.topk(latent_vars, k=constrs_num)[-1].cpu().detach().numpy()
            inference_time.append(time.time()-start_time)

            VBas = np.array([-1] * var_num)
            CBas = np.array([-1] * constrs_num)
            VBas[pred_indices] = 0
            vbas = VBas + 1
            cbas = CBas + 1
            #f = open(highs_path+file+'.bas','w')
            #f.write("HiGHS v1\nValid\n# Columns "+str(len(vbas))+"\n"+' '.join(map(str, vbas))+"\n# Rows "+str(len(cbas))+"\n"+' '.join(map(str, cbas)))
            #f.close()
        del latent_vars
        del model
        torch.cuda.empty_cache()
        '''np.save(highs_path+"QR_"+file, np.array(QR_time))
        np.save(highs_path+"dot_"+file, np.array(dot_time))
        np.save(highs_path+"inf_"+file, np.array(inference_time))
        np.save(highs_path+"tol_"+file, np.array(QR_time) + np.array(dot_time) + np.array(inference_time))
'''
print("final eval result-------------------------------------------")
print(f'len={file_num}, avg_f1={avg_f1 / file_num}, avg_acc={avg_acc / file_num}, time={time.time() - tol_time}')
