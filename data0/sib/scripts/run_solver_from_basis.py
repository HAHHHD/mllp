import os, sys

sys.path.insert(0, os.path.dirname(__file__) + '/../')
from utils import *
import glob
import os
import os.path as osp
import utils
from multiprocessing import Pool
from functools import partial
from val import InferenceManager
import numpy as np

def run_highs_from_basis(path, mps, 
                    basis_folder='pred-basis', log_folder='highs-gnn-bas', 
                    method=1, 
                    **kwargs): 
    LP=extract_fn(mps) 
    MPS = f'{path}/mps/{LP}.mps'
    if not osp.exists(MPS): shell(f'gzip -d -k -f {MPS}.gz')
    assert osp.exists(MPS), MPS   

    BAS = f'{path}/{basis_folder}/{LP}.bas' 
    assert osp.exists(BAS), BAS
    # todo put BAS3 m!=1 to other folders
    BAS3 = f'{path}/opt-from-{basis_folder}/{LP}.bas'
    mkdir_p(f'{osp.dirname(BAS3)}') 
    print(method)
    #cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 --time_limit 600 -ss {method} ' 
    #out,err = shell(cmd, verbose=1) 

    cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 --time_limit 600 -bi {BAS} -bo {BAS3} -ss {method} ' 
    out,err = shell(cmd, verbose=1) 
    if method!=1: log_folder+=f'-m{method}' 
    shell(f'mkdir -p {path}/log/{log_folder}')
    if log_folder: 
        with open(f'{path}/log/{log_folder}/{LP}.log', 'w') as f:  
            f.write(out)
    return extract_time(out)


if __name__ == "__main__":
    args = utils.Environment(
        )
    args.double_check(key='lp_method')
    inf_mng=eval(args.inference_manager) 
    log_folder = inf_mng.get_log_folder() 
    shell(f'mkdir -p {args.log_dir}/log/{log_folder}') 
    shell(f'rmdir {args.log_dir}/mps') 
    shell(f'ln -sf {args.dataset_prefix}/mps {args.log_dir}', verbose=1)
    shell(f'cp -r {args.dataset_prefix}/log/{args.solver_prefix}* {args.log_dir}/log/', verbose=1)
    path = args.log_dir
    tgt=f'{args.log_dir}/time.h5'
    from dataset import LPDataset, MyToBipartite 
    ds = LPDataset(args.dataset_processed_prefix.replace('-m4',''),#.replace('-m2',''), 
                   chunk=None, transform=MyToBipartite(phase='val')
                   )
    ds.dump_size_info(tgt) 
    df=df_load(tgt) 
    if args.split=='val':
        logging.info('limit run_from_basis to val only ')
        fns = df.loc[df.split==args.split, 'fn']   
    else:
        assert args.split=='trainval' 
        fns = df['fn']

    iters = []
    times = []
    func = run_highs_from_basis 
    for fn in fns: 
        gW=args.gW 
        iter, time = func(path, fn, 
                basis_folder=inf_mng.get_basis_folder(), 
                log_folder=inf_mng.get_log_folder(),             
                verbose=args.verbose, method=args.lp_method, gW=gW
            )
        iters.append(iter)
        times.append(time)
    np.save("./time/miplibCs_miplibCs_primal_iter", np.array(iters))
    np.save("./time/miplibCs_miplibCs_primal_time", np.array(times))
    