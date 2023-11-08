import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../')
import pandas as pd
from dataset import LPDataset, MyToBipartite
from utils import *

def map_back_to_baseline(log_fdl_nm, solver_pref): 
    '''
    example: 
    gnn-bas.*m2 -> {SV}no-bas-m2 
    gnn-bas.*m4 -> SV-no-bas-m4
    gnn-bas.* -> SV-no-bas 
    SV-ca-bas-m2 -> SV-no-bas-m2 
    SV-ca-bas -> SV-no-bas 
    else '' 
    '''
    for method in ['m2', 'm4', '']: 
        sfx = f'-{method}' if method !='' else '' 
        if re.match(f'.*[(gnn)|(ca)]-bas.*{method}',log_fdl_nm): 
            return f'{solver_pref}no-bas{sfx}' 
    return '' 

if __name__ == '__main__':
    args = Environment(
        exp_nm='highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3',
        solver_prefix='highs-', 
        dataset='mirp', 
    )
    mkdir_p(args.log_dir) 
    shell(f'rmdir {args.log_dir}/mps') 
    shell(f'rm {args.log_dir}/mps') 
    shell(f'ln -sf {args.dataset_prefix}/mps {args.log_dir}/', verbose=1)
    shell(f'mkdir -p {args.log_dir}/log') 
    shell(f'cp -r {args.dataset_prefix}/log/* {args.log_dir}/log/', verbose=1) 
    shell(f'cp -r {args.log_dir}/log/highs-ca-bas-m1 {args.log_dir}/log/highs-ca-bas') 
    
    dump_info_pn = f'{args.log_dir}/time.h5' 
    if osp.exists(dump_info_pn): 
        df = df_load(dump_info_pn) 
    else:
        ds = LPDataset(args.dataset_processed_prefix,
                   chunk=None, transform=MyToBipartite(phase='val')
                   )
        df = ds.dump_size_info(dump_info_pn) 
    
    for fname in os.listdir(f'{args.log_dir}/mps/'): 
        fn = extract_fn(fname) 
        fname = fn+'.log' 
        optional_paths = os.listdir(f'{args.log_dir}/log/') 
        for optinal_path in optional_paths: 
            fdl_pref=parse_str(optinal_path, 'solver_prefix')
            if fdl_pref and fdl_pref!=args.solver_prefix: continue 
            tgt=f'{args.log_dir}/log/{optinal_path}/{fname}'     
            if osp.exists(tgt):    
                iters, tm = extract_time(tgt) 
                df.loc[df.fn==fn, f'{optinal_path}/niter']=iters 
                df.loc[df.fn==fn, f'{optinal_path}/time']=tm 
                if 'gnn' in optinal_path: 
                    num_basis, n_slack_basis_begin, n_add_slack = extract_nrepair(tgt)
                    repair_p1 = n_add_slack / num_basis 
                    n_var_basis_begin = (num_basis - n_slack_basis_begin) 
                    if n_var_basis_begin == 0: repair_p2=np.inf 
                    else: repair_p2 = n_add_slack / n_var_basis_begin 
                    df.loc[df.fn==fn, f'{optinal_path}/repair_p']=repair_p1 
                    df.loc[df.fn==fn, f'{optinal_path}/repair_p2']=repair_p2 

                    fact_time = extract_fact_time(tgt) 
                    df.loc[df.fn==fn, f'{optinal_path}/fact_time']=fact_time 
      
    for method in df.columns: 
        bs_method = map_back_to_baseline(method, args.solver_prefix) 
        if not bs_method: continue   
        method, impr = method.split('/') 
        if impr not in ['time', 'niter']: continue 
        df[f'{method}/impr_{impr}'] = -(df[f'{method}/{impr}'] -
                                    df[f'{bs_method}/{impr}']) / df[f'{bs_method}/{impr}']
        

    if 'fact_time' in df: df['fact_time']=df['fact_time'].astype('float')
    
    cols = filter_cols(['split', 
                'acc', 'prec', 'recl', 
                ] , df.columns )
    
    cols+=list(filter( 
                    lambda x: ('/niter' in x or '/time' in x) , 
                        df.columns,
                    ) )
    print(df[cols].groupby('split').agg(['mean', 'std']).T) 
    print(df[cols].groupby('split').get_group('val').describe().loc[['mean','std']].apply(proc, axis=0).to_frame())
    df_dump(df, dump_info_pn) 
