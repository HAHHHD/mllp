from utils import * 

def run_app(args):
    ARGS=args.cvt_to_str() 
    # if not '-to-' in args.exp_nm and not 'tmp' in args.exp_nm: args.double_check() 
    logging.info(f'args {ARGS}')
    EXP_LD_PATH=args.log_dir #f'/cache/runs/{exp_nm}/'

    args.clear_terminal() 
    # shell_sp(f'python scripts/cvt_to_pkl.py {ARGS} ', verbose=1) # todo
    shell_sp(f'python dataset.py {ARGS} --skip_exist 0 ', verbose=1) 
    shell_sp(f'python train.py {ARGS} ', verbose=1) ## todo 
    # shell_sp(f'rm -f {args.log_dir}/time.h5') 
    shell_sp(f'python scripts/pred_basis.py {ARGS} --load_from {EXP_LD_PATH}/mdl.pth ') ## output: pred-basis, time in time.h5 
    shell(f'cp -r {args.dataset_prefix}/log/{args.solver_prefix}* {args.log_dir}/log/', verbose=1)
    lp_method=args.lp_method # 'cg'
    shell_sp(f'python scripts/run_solver_from_basis.py {ARGS} --num_workers 1 --lp_method {lp_method} ')  ## possible: load pred basis, Numerical error -> no bas file saved ## output: opt-from-pred-basis log/gnn-bas-0 
    shell_sp(f'python val.py {ARGS} --load_from {EXP_LD_PATH}/mdl.pth ') ## opt-from-pred-basis -> time.h5 # ## may merge them 
    shell_sp(f'python scripts/extract_time.py {ARGS} ')


if __name__=='__main__':
    args=Environment(
        dataset='mirp', #'stoch-sc-5', #
        arch='GCN_FC(8,8,hids=1024,depth=3)', #'GCN_FC(8,8,hids=1024,depth=5)', #
        solver_prefix='highs-', 
        loss='focal', 
        exp_nm='tmp',
        epochs=40, dev=0, skip_exist=0, verbose=1, num_workers=8,
        inference_manager='InferenceManager(0)', #gW=.15, 
        split='val', # trainval
        ) 
    run_app(args) 
