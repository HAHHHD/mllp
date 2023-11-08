import torch
from pathlib import Path
import re
import urllib.request
import random
import os
import numpy as np
import scipy
import time

def get_random_dataset(num_items, num_sets, seed):
    random.seed(seed)
    dataset = []
    for i in range(100):
        weights = [random.randint(1, 100) for _ in range(num_items)]
        sets = []
        for set_idx in range(num_sets):
            covered_items = random.randint(10, 30)
            sets.append(random.sample(range(num_items), covered_items))
        dataset.append((f'rand{i}', weights, sets))
    return dataset

def get_netlib_dataset_dense(normalize=True):
    path = "processed/mpsM"
    files = os.listdir(path)
    dataset = []
    train_dict = {}
    train_dict["obj"] = []
    file_num = 0
    for file in files:
        if file == '1.mps':
            continue
        if file_num >= 1000:
            break
        if normalize:
            file_path = "dataset/mps0000s/" #"dataset/netlib_mps_norm/"
        else:
            file_path = "dataset/netlib_mps/"
        #v_basis_list = np.load(file_path+file+"_v.npy")
        #c_basis_list = np.load(file_path+file+"_c.npy")
        #basis_opt = np.concatenate([v_basis_list, c_basis_list])
        #basis_opt = np.load(file_path+file+"_basis.npy")
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
        if constrs_matrix.shape[0] < 30000000:
            print("---------------------Instance {} size: {}".format(file_num, constrs_matrix.shape))
            start = time.time()
            #Q = constrs_matrix.tocsc() * 1e3
            constrs_matrix *= 1e3
            coefs *= 1e3
            if constrs_matrix.shape[0] > 40000:
                constrs_matrix = constrs_matrix.tocsc()[:, :1200000000 // constrs_matrix.shape[0]].tocoo()
            #constrs_matrix = constrs_matrix.todense()
            print("QR00 time:", time.time() - start)
            start = time.time()
            '''for i in range(coefs.shape[0]):
                if abs(coefs[i]) < 1e-6:
                    if np.linalg.norm(constrs_matrix[i]) > 1e-6:
                        constrs_matrix[i] /= np.linalg.norm(constrs_matrix[i])
                    coefs[i] = 0
                else:
                    constrs_matrix[i] /= abs(coefs[i])
                    coefs[i] = np.sign(coefs[i])
                    tol_norm += np.sum(np.square(constrs_matrix[i]))
            for i in range(coefs.shape[0]):
                if coefs[i] == 0:
                    constrs_matrix[i] /= np.sqrt(tol_norm)'''
            print("QR01 time:", time.time() - start)
            start = time.time()
            values = constrs_matrix.data
            indices = np.vstack((constrs_matrix.row, constrs_matrix.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = constrs_matrix.shape
            Q = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to("cuda")
            Q = Q.to_dense()
            torch.cuda.empty_cache()
            #Q = torch.cuda.FloatTensor(constrs_matrix)
            #norm = scipy.sparse.linalg.norm(Q, axis=1).reshape(-1, 1)
            norm = torch.linalg.norm(Q, axis=1)
            #print("QR02 time:", time.time() - start)
            #start = time.time()
            #Q = Q / norm
            #coefs = (coefs.reshape(-1, 1) / norm).squeeze(-1)
            tmp = (abs(norm) - 1e-6) < 0
            norm += tmp * 1e8
            Q /= norm.unsqueeze(0).T
            coefs = (np.expand_dims(coefs, axis=0).T / norm.unsqueeze(0).T.to('cpu').numpy()).T.squeeze()
            '''for i in range(coefs.shape[0]):
                if norm[i] > 1e-6:
                    coefs[i] /= norm[i]
                    Q[i] /= norm[i]'''
                #else:
                    #print("111opps!!!!!!!!!!!!!!!!!!")
            coefs /= np.linalg.norm(coefs)
            #print(Q)
            print("QR1 time:", time.time() - start)
            start = time.time()
            del norm
            torch.cuda.empty_cache()
            #Q *= 1e3
            #norm = scipy.sparse.linalg.norm(Q, axis = 0).reshape(-1, 1)
            '''for i in range(Q.shape[1]):
                if norm[i] > 1e-6:
                    Q[:, i] /= norm[i]
                else:
                    print("222opps!!!!!!!!!!!!!!!!!!")'''
            #Q = (Q.T / norm).T
            #del norm
            #torch.cuda.empty_cache()
            Q, _ = torch.linalg.qr(Q)
            print("QR2 time:", time.time() - start)
            start = time.time()
            del _
            torch.cuda.empty_cache()
            Q = Q.cpu().numpy()
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
        else: 
            #constrs_matrix = scipy.sparse.coo_matrix(constrs_matrix.T)
            print("QR1 time:", time.time() - start)
            start = time.time()
            print(constrs_matrix, constrs_matrix.shape)
            print(len(scipy.sparse.find(constrs_matrix)[0]))
            Q, R, E, rank = sparseqr.qr( constrs_matrix, economy = True)
            print(len(scipy.sparse.find(Q)[0]), len(scipy.sparse.find(R)[0]))
            print("QR2 time:", time.time() - start)
            start = time.time()
            Q = Q.todense()
        print(Q.shape)
        #print(constrs_matrix)
        #print(Q, coefs)
        print("QR3 time:", time.time() - start)
        dataset.append((file, Q, coefs, basis_opt))
        train_dict[file] = []
    return dataset, train_dict


def get_netlib_dataset(normalize=True):
    path = "netlib_mps"
    files = os.listdir(path)
    dataset = []
    train_dict = {}
    train_dict["obj"] = []
    for file in files:
        if normalize:
            file_path = "dataset/netlib_mps_norm/"
        else:
            file_path = "dataset/netlib_mps/"
        #v_basis_list = np.load(file_path+file+"_v.npy")
        #c_basis_list = np.load(file_path+file+"_c.npy")
        #basis_opt = np.concatenate([v_basis_list, c_basis_list])
        basis_opt = np.load(file_path+file+"_basis.npy")
        coefs = np.load(file_path+file+"_coefs.npy")
        rhs = np.load(file_path+file+"_rhs.npy")
        constrs_sp_matrix = scipy.sparse.load_npz(file_path+file+"_constrs.npz")
        constrs = np.split(constrs_sp_matrix.indices, constrs_sp_matrix.indptr)[1:-1]
        constrs_weights = constrs_sp_matrix.data
        dataset.append((file, constrs, constrs_weights, coefs, rhs, basis_opt))
        train_dict[file] = []
    return dataset, train_dict

def get_facebook_dataset():
    dataset = []
    center_nodes = {
        'facebook': [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980],
    }
    for platform in center_nodes.keys():
        with open(f'data/social_net/{platform}_combined.txt') as f:
            edges = []
            node_ids = set()
            for e in f.readlines():
                e_str = e.strip().split()
                n1, n2 = int(e_str[0]), int(e_str[1])
                edges.append((n1, n2))
                node_ids.add(n1)
                node_ids.add(n2)
        id_map = {n: i for i, n in enumerate(node_ids)}
        weights = [1 for _ in node_ids]
        sets = [[] for _ in node_ids]
        for n1, n2 in edges:
            if n1 in center_nodes[platform] or n2 in center_nodes[platform]:
                continue
            sets[id_map[n1]].append(id_map[n2])
            sets[id_map[n2]].append(id_map[n1])
        dataset.append((platform, weights, sets))
    return dataset


def get_twitch_dataset():
    import math
    dataset = []
    languages = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
    for language in languages:
        with open(f'data/twitch/{language}/musae_{language}_edges.csv') as f:
            edges = []
            node_ids = set()
            for e in f.readlines():
                e_str = e.strip().split(',')
                if e_str[0] == 'from' and e_str[1] == 'to':
                    continue
                n1, n2 = int(e_str[0]), int(e_str[1])
                edges.append((n1, n2))
                node_ids.add(n1)
                node_ids.add(n2)
        id_map = {n: i for i, n in enumerate(node_ids)}
        weights = [-1 for _ in node_ids]
        with open(f'data/twitch/{language}/musae_{language}_target.csv') as f:
            for line in f.readlines():
                line_str = line.strip().split(',')
                if line_str[0] == 'id':
                    continue
                weights[id_map[int(line_str[5])]] = math.floor(math.log(int(line_str[3]) + 1))
        assert min(weights) >= 0
        sets = [[] for _ in node_ids]
        for n1, n2 in edges:
            sets[id_map[n1]].append(id_map[n2])
        dataset.append((language, weights, sets))
    return dataset

ONLINE_REPO = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/'


problem_set = {
    'scp4': 10,
    'scp5': 10,
    'scp6': 5,
    'scpa': 5,
    'scpb': 5,
    'scpc': 5,
    'scpd': 5,
    'scpe': 5,
    'scpnre': 5,
    'scpnrf': 5,
    'scpnrg': 5,
    'scpnrh': 5
}


class SCP_ORLIB:
    def __init__(self, fetch_online=False):
        super(SCP_ORLIB, self).__init__()
        self.classes = problem_set.keys()
        self.data_list = []
        self.data_path = Path('data/scp_orlib')

        for cls in self.classes:
            cls_len = problem_set[cls]
            for i in range(cls_len):
                self.data_list.append(cls + '{}'.format(i + 1))

        # define compare function
        def name_cmp(a, b):
            a = re.findall(r'[0-9]+|[a-z]+', a)
            b = re.findall(r'[0-9]+|[a-z]+', b)
            for _a, _b in zip(a, b):
                if _a.isdigit() and _b.isdigit():
                    _a = int(_a)
                    _b = int(_b)
                cmp = (_a > _b) - (_a < _b)
                if cmp != 0:
                    return cmp
            if len(a) > len(b):
                return -1
            elif len(a) < len(b):
                return 1
            else:
                return 0

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # sort data list according to the names
        self.data_list.sort(key=cmp_to_key(name_cmp))
        print(self.data_list)

        fetched_flag = self.data_path / 'fetched_online'

        if fetch_online or not fetched_flag.exists():
            self.__fetch_online()
            fetched_flag.touch()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Notice: the indices start from 0 which is different from the original ORLIB format (start from 1)"""
        name = self.data_list[idx]

        dat_path = self.data_path / (name + '.txt')
        dat_file = dat_path.open()

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]

        nrows, ncols = dat_list[0]

        # read data
        row_idx = 1

        # read column weights
        column_weights = []
        while len(column_weights) < ncols:
            column_weights += dat_list[row_idx]
            row_idx += 1

        assert len(column_weights) == ncols

        # read row sets
        row_sets = []
        remain_len_of_this_row = 0
        while row_idx < len(dat_list):
            if remain_len_of_this_row == 0:
                assert len(dat_list[row_idx]) == 1
                remain_len_of_this_row = dat_list[row_idx][0]
                row_sets.append([])
            else:
                row_sets[-1] += [item-1 for item in dat_list[row_idx]]  # we let the index start from 0
                remain_len_of_this_row -= len(dat_list[row_idx])
                assert remain_len_of_this_row >= 0
            row_idx += 1

        return name, column_weights, row_sets

    def __fetch_online(self):
        """
        Fetch from online QAPLIB data
        """
        for name in self.data_list:
            dat_content = urllib.request.urlopen(ONLINE_REPO + '{}.txt'.format(name)).read()

            dat_file = (self.data_path / (name + '.txt')).open('wb')
            dat_file.write(dat_content)
