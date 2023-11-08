import os, sys

sys.path.insert(0, os.path.dirname(__file__) + '/../')
from utils import *
import glob
import os
import os.path as osp
import utils
from multiprocessing import Pool
from functools import partial

def run_highs(path, mps, skip_exist, method=4, 
            *args, **kwargs):
    #for highs: method=1 dual; method=4 primal 
    LP=extract_fn(mps) 
    MPS=f'{path}/mps/{LP}.mps' 
    if method ==1: 
        BAS=f'{path}/highs-basis/{LP}.bas' 
        LOG=f'{path}/log/highs-no-bas/{LP}.log' 
    else: 
        BAS=f'{path}/highs-basis-m{method}/{LP}.bas' 
        LOG=f'{path}/log/highs-no-bas-m{method}/{LP}.log' 
    mkdir_p(f'{osp.dirname(BAS)}')
    mkdir_p(f'{osp.dirname(LOG)}')
    if skip_exist and osp.exists(LOG) and (not BAS or osp.exists(BAS)): 
        print('skip:', LOG, BAS)
        return
    if not osp.exists(MPS): shell(f'gzip -d -k -f {MPS}.gz', verbose=1)
    assert osp.exists(MPS), MPS
    cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 -bo {BAS} -ss {method} ' 
    out, _ = shell(cmd) 
    iters, times = extract_time(out) 
    print(out) 
    print(LP, iters, times) 
    with open(LOG, 'w') as f: 
        f.write(out)
    

if __name__ == "__main__":
    args = utils.Environment( )
    path = args.dataset_prefix 
    fns = glob.glob(f'{path}/mps/*.mps') 
    if len(fns)==0: fns = glob.glob(f'{path}/mps/*.mps.gz') 
    ## above assume gzip -d atomic for all *.gz 
    fns = sorted(fns, key=lambda nm: (len(nm), nm))
    fns = split_out_chunk(fns, args.chunk) 
    if not args.skip_exist: shell(f'rm -r {args.dataset_prefix}/log/*', verbose=1) 

    ## only test with 1 core 
    for fn in fns: 
       if args.solver_prefix=='highs-':
            from scripts.run_solver_from_basis import run_highs_from_basis 
            
            run_highs(path, fn, args.skip_exist) 
            # For anoymous, the crash method is provided yet, but we provide the initial basis of crash method for mirp. The whole codebase will be released later 
            #run_highs_from_basis(path, fn, 'highs-ca-init-bas-m1', 'highs-ca-bas-m1') 

            # Test with primal method will be provided in the whole codebase 