import os, sys

sys.path.insert(0, os.path.dirname(__file__) + '/../')
from utils import *
import utils
import numpy as np
from scipy.sparse import csr_matrix


def index_of_y_in_x(y, x):
    assert np.isin(y, x).all()
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    return index[sorted_index]

def drop_interity(file: str, tgt: str): 
    # todo use .relax api 
    from mip import Model
    mdl = Model(solver_name='CBC')
    # blker=printBlocker()
    # blker.blk() 
    if file.endswith('.gz'):
        out,err=shell(f'gzip -d -f -k {file}') 
        if err!='': 
            logging.error(f' zip fail {err}')
            return 
        file = file.replace('.gz', '')
    try: 
        mdl.read(file)
    except: 
        logging.error(f'read fail') 
        return 
    # blker.unblk() 
    # model parameters
    num_vars = len(mdl.vars)
    num_cons = len(mdl.constrs)

    # variable types and bounds
    ## If no bounds are specified, CPLEX will automatically set the lower bound to 0 and the upper bound to +∞.
    lb = np.zeros(num_vars)
    ub = np.empty(num_vars)
    types = np.empty(num_vars, dtype=str)
    for i, var in enumerate(mdl.vars):
        lb[i] = var.lb
        ub[i] = var.ub
        types[i] = var.var_type
        if var.var_type=='B': assert var.lb==0 and var.ub==1
        var.var_type='C' 
    
    mkdir_p(osp.dirname(tgt)) 
    tgt=tgt.replace('.mps.gz', '.mps')
    mdl.write(tgt) 
    shell_sp(f'mv {tgt}.mps.gz {tgt}.gz',) 
    cplex_read_write_mps(f'{tgt}.gz')  

def cplex_read_write_mps(fn):
    cmd=f'cplex -c "read {fn}" "write {fn}" "y" ' 
    shell(cmd, verbose=1) 

def read_mps(file: str, only_names=False):
    """
    ref: https://stackoverflow.com/questions/73266140/how-to-read-an-mps-file-for-scipy-milp
    Reads a .mps and saves all the data of the MILP:

    min c^T * x

    s.t. b_l <= A*x <= b_u
          lb <=   x <= ub
    """
    from mip import Model
    mdl = Model(solver_name='CBC')
    # blker=printBlocker()
    # blker.blk() 
    if file.endswith('.gz'):
        shell(f'gzip -d -f -k {file}')
        mps_file = file.replace('.gz', '')
        mdl.read(mps_file)
        # shell(f'rm {mps_file}')
    else:
        mdl.read(file)
    # blker.unblk() 
    # model parameters
    num_vars = len(mdl.vars)
    num_cons = len(mdl.constrs)

    # variable types and bounds
    ## If no bounds are specified, CPLEX will automatically set the lower bound to 0 and the upper bound to +∞.
    lb = np.zeros(num_vars)
    ub = np.empty(num_vars)
    for i, var in enumerate(mdl.vars):
        assert var.idx==i
        lb[i] = var.lb
        ub[i] = var.ub
        assert var.var_type=='C' 
    
    # objective
    var_nms = np.array([var.name for var in mdl.vars])
    con_nms = np.array([con.name for con in mdl.constrs])
    if only_names: return con_nms, var_nms
    var_nm_to_c = {}
    for var, weight in mdl.objective.expr.items():
        var_nm_to_c[var.name] = weight
    c = np.vectorize(lambda var: var_nm_to_c.get(var, 0))(var_nms)
    if mdl.sense != "MIN": c *= -1.0

    # constraint coefficient matrix
    b_l = -np.inf * np.ones(num_cons)
    b_u = np.inf * np.ones(num_cons)
    for i, con in enumerate(mdl.constrs):
        assert con.idx==i 
        if con.expr.sense == "=":
            b_l[i] = con.rhs
            b_u[i] = con.rhs
        elif con.expr.sense == "<":
            b_u[i] = con.rhs
        elif con.expr.sense == ">":
            b_l[i] = con.rhs
        else: 
            raise ValueError(f'unkn {con.expr.sense}')
    row_ind = []
    col_ind = []
    data = []
    for i, con in enumerate(mdl.constrs):
        expr = con.expr.expr
        for var, coeff in expr.items():
            row_ind.append(con.name)
            col_ind.append(var.name)
            data.append(coeff)
    row_ind = index_of_y_in_x(row_ind, con_nms)
    col_ind = index_of_y_in_x(col_ind, var_nms)
    A = csr_matrix((data, (row_ind, col_ind)), shape=(num_cons, num_vars))
    return c, b_l, A, b_u, lb, ub, con_nms, var_nms


def read_leaves(cons, fn=''):
    status_str_to_int = {'LL': 0, 'BS': 1, 'UL': 2}
    lbls = np.empty(len(cons), dtype=int)
    duals = np.empty(len(cons), dtype=int)
    slacks = np.empty(len(cons), dtype=int)
    nms = np.empty(len(cons), dtype=object)
    for con in cons:
        ind, sta = con.attrib['index'], con.attrib['status']
        sta = status_str_to_int[sta]
        if 'dual' in con.attrib:
            dual = float(con.attrib['dual'])
            duals[int(ind)] = dual
            slacks[int(ind)] = float(con.attrib['slack'])
            # if dual < -1e-8 and sta == 0:
            #     sta = 2 - sta
            #     # if fn: print(fn)
        lbls[int(ind)] = sta
        nms[int(ind)] = (con.attrib['name'])
    return lbls, nms


def read_sol(fn):
    import xml.etree.ElementTree as ET
    tree = ET.parse(fn)
    cons = tree.getroot()[2]
    con_lbls, con_nms = read_leaves(cons, fn)
    varibles = tree.getroot()[3]
    var_lbls, var_nms = read_leaves(varibles)
    return con_lbls, var_lbls, con_nms, var_nms

def read_bas_highs(fn):
    assert osp.exists(fn),fn
    def parse_line(line):
        l=line.split() 
        return np.array(l, 'int') 
    with open(fn,'r') as f: 
        lines=f.readlines() 
    for idx, line in enumerate(lines):
        if 'Columns' in line: 
            var_stas=parse_line(lines[idx+1]) 
        if 'Rows' in line: 
            con_stas=parse_line(lines[idx+1]) 
    return con_stas, var_stas 

def read_bas(fn, con_nms=None, var_nms=None): 
    status_str_to_int = {'LL': 0, 'BS': 1, 'UL': 2,
                         'XU': (1, 2), 'XL': (1, 0),
                         }
    con_label_default, var_label_default = 1, 0
    with open(fn,'r') as f:
        lines = f.readlines()
    if 'HiGHS' in lines[0]: return read_bas_highs(fn) 
    assert con_nms is not None
    con_name_to_label, var_name_to_label = {}, {}
    for line in lines:
        # if '* ENCODING' in line: continue 
        # if 'ENDATA' in line: continue 
        line = line.split()
        status = line[0]
        if status not in status_str_to_int: continue 
        if status in ['XU', 'XL']:
            status = status_str_to_int[status]
            var_name, con_name = line[1:]
            var_name_to_label[var_name] = status[0]
            con_name_to_label[con_name] = status[1]
        else:
            status = status_str_to_int[status]
            var_name = line[1]
            var_name_to_label[var_name] = status
    return (np.vectorize(lambda nm: con_name_to_label.get(nm, con_label_default))(con_nms),
            np.vectorize(lambda nm: var_name_to_label.get(nm, var_label_default))(var_nms))

def check_lb(lbls, l, nms): 
    idx = np.where((l==-np.inf)&(lbls==0))[0]
    if len(idx)!=0: 
        logging.error(f'#violate {len(idx)} {nms[idx[0]]}') 
        lbls[idx]=2

def check_ub(lbls, u, nms): 
    idx = np.where((u==np.inf)&(lbls==2))[0] 
    if len(idx)!=0: 
        logging.error(f'#voliate {len(idx)} {nms[idx[0]]}') 
        lbls[idx]=0

def cvt_one(nm, skip_exist):
    logging.info(f'proc {nm}') 
    prefix = osp.normpath(osp.dirname(nm) + '/../')
    base_nm = extract_fn(nm)
    src = f'{prefix}/mps/{base_nm}.mps'
    dst = f'{prefix}/{args.solver_prefix}inp_tgt{args.get_method_sfx()}/raw/{base_nm}.pk'
    assert osp.exists(src), src

    bas_fn = f'{prefix}/{args.solver_prefix}basis{args.get_method_sfx()}/{base_nm}.bas'
    if not osp.exists(bas_fn): 
        logging.error(f'no {bas_fn}')
        return 

    if skip_exist and osp.exists(dst): return
    c, b_l, A, b_u, l, u, con_nms, var_nms = read_mps(src)
    b_u[b_u > 1e308] = np.inf
    b_l[b_l < -1e308] = -np.inf
    u[u > 1e308] = np.inf
    l[l < -1e308] = -np.inf

    # con_lbls2, var_lbls2, con_nms2, var_nms2 = read_sol(f'{prefix}/sol/{base_nm}.sol')
    con_lbls, var_lbls = read_bas(bas_fn, con_nms, var_nms)
    # assert (con_lbls == con_lbls2).all()
    # assert (var_lbls == var_lbls2).all()
    # assert (con_nms == con_nms2).all()
    # assert (var_nms == var_nms2).all()

    check_lb(var_lbls, l, var_nms) 
    check_ub(var_lbls, u, var_nms) 

    # idx = np.where((b_l == -np.inf) & (con_lbls == 0))[0]
    # if len(idx) != 0: print('-- error', con_lbls[idx], base_nm) 
    # con_lbls[idx] = 2

    # idx = np.where((b_u == np.inf) & (con_lbls == 2))[0]
    # if len(idx) != 0: print('++ error', con_lbls[idx], base_nm) 
    # con_lbls[idx] = 0
    assert (con_lbls[b_l == -np.inf] != 0).all()
    assert (con_lbls[b_u == np.inf] != 2).all()
    ## Why cplex change constraint direction: --> standard form

    A = A.tocoo()
    p = Thread(
        target=utils.msgpack_dump,
        args=([c, b_l, (A.row, A.col, A.data), b_u, l, u,
               con_lbls, var_lbls, con_nms, var_nms
               ], dst),
        daemon=True,
    )
    p.start()
    p.join()
    # [c, b_l, (row, col, data), b_u, l, u,
    #          con_lbls, var_lbls, con_nms, var_nms] = msgpack_load(dst, copy=True)
    # ncons, nvars = len(con_nms), len(var_nms)
    # from scipy.sparse import coo_matrix
    # A2 = coo_matrix((data, (row, col)), shape=(ncons, nvars))
    # A2=A2.tocsr() # should be same with A 


if __name__ == '__main__':
    from utils import Environment

    args = Environment(
    )

    prefix = args.dataset_prefix
    shell(f'mkdir -p {prefix}/{args.solver_prefix}inp_tgt{args.get_method_sfx()}/raw/')
    fns = glob.glob(f'{prefix}/mps/*.mps') 
    if len(fns)==0: fns = glob.glob(f'{prefix}/mps/*.mps.gz') 
    ## above assume gzip -d atomic for all *.gz 
    fns = sorted(fns, key=lambda nm: (len(nm), nm))
    if len(fns)==0: raise ValueError('found no mps or gz')
    num_workers = args.num_workers
    if num_workers > 1:
        with Pool(num_workers) as pool:  # ThreadPoolExecutor(max_workers=48):
            pool.map(partial(cvt_one, skip_exist = args.skip_exist), fns)
    else:
        for fn in fns:
            cvt_one(fn, args.skip_exist)

