from utils import * 

def run_app(args):
    ARGS=args.cvt_to_str() 
    shell_sp(f'python scripts/run_solver.py {ARGS} --num_workers 1 ', verbose=1)
    shell_sp(f'python scripts/cvt_to_pkl.py {ARGS} ', verbose=1, raise_error=1) # todo chunk 
    shell_sp(f'python dataset.py {ARGS} --skip_exist 0 ', verbose=1, raise_error=1) 

if __name__=='__main__':
    args=Environment(dataset='mirp', 
                solver_prefix='highs-', verbose=1, chunk='0/1', 
                num_workers=8, exp_nm='tmp', dev=0, skip_exist=1,
    ) 
    run_app(args) 
