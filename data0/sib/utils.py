
import copy, os
import numpy as np
import pandas as pd
import torch, re
import sys, os, time, logging, glob

try:
    import seaborn as sns 
except: 
    os.system('pip install seaborn') 
    import seaborn as sns 
import torch_geometric
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import os.path as osp
from tqdm import tqdm
from functools import partial 

logging.root.setLevel(logging.INFO)
np.set_string_function(lambda arr: f'np {arr.shape} {arr.dtype} '
                                   f'{arr.__str__()} '
                                   f'dtype:{arr.dtype} shape:{arr.shape} np', repr=True)


class Timer(object):
    def __init__(self, print_tmpl=None, start=True, ):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self, aux=''):
        """Total time since the timer is started.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        self._t_last = time.time()
        logging.info(f'{aux} time {self.print_tmpl.format(self._t_last - self._t_start)}')
        return self._t_last - self._t_start

    def since_last_check(self, aux='', verbose=True):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking operation.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        if verbose and aux:
            logging.info(f'{aux} time {self.print_tmpl.format(dur)}')
        return dur

def shell_sp(cmd, raise_error=True, **kwargs):
    import os, logging, subprocess
    print('-'*30)
    logging.info('cmd:' + cmd) 
    res= subprocess.call(cmd.split())  
    if res!=0: 
        msg = f' fail code: {res}'
        if raise_error:
            raise ValueError(msg) 
        else:
            logging.error(msg) 
    return res 

def shell(cmd, block=True, return_msg=True, verbose=True, timeout=None):
    import os, logging, subprocess
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    if 'anaconda' not in my_env['PATH']:
        my_env['PATH'] = home + "/anaconda3/bin/:" + my_env['PATH']
    my_env['PATH'] = home + '/bin/:' + my_env['PATH']
    # my_env['http_proxy'] = ''
    # my_env['https_proxy'] = ''
    if verbose:
        logging.info('cmd is ' + cmd)
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        if return_msg:
            msg = task.communicate(timeout)
            msg = [msg_.decode('utf-8') for msg_ in msg]
            if msg[0] != '' and verbose:
                logging.info('stdout | {}'.format(msg[0]))
            if msg[1] != '' and verbose:
                logging.error(f'stderr | {msg[1]} | cmd {cmd}')
            return msg
        else:
            return task
    else:
        logging.debug('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        return task


def mkdir_p(path):
    import os
    path = str(path)
    if path == '': return
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def pickle_dump(data, file, **kwargs):
    import pickle, pathlib
    # python2 can read 2
    kwargs.setdefault('protocol', pickle.HIGHEST_PROTOCOL)
    if isinstance(file, str) or isinstance(file, pathlib.Path):
        mkdir_p(osp.dirname(file))
        print('pickle into', file)
        with open(file, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(data, file, **kwargs)
    else:
        raise TypeError("file must be str of file-object")


def pickle_load(file, **kwargs):
    import pickle
    if isinstance(file, str):
        with open(file, 'rb') as f:
            data = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        data = pickle.load(file, **kwargs)
    return data


def json_dump(obj, file, mode='a'):  # write not append!
    # import codecs
    import json
    if isinstance(file, str):
        # with codecs.open(file, mode, encoding='utf-8') as fp:
        with open(file, 'w') as fp:
            json.dump(obj, fp, sort_keys=True, indent=4
                      # ensure_ascii=False
                      )
    elif hasattr(file, 'write'):
        json.dump(obj, file)


def json_load(file):
    import json
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def msgpack_dump(obj, file, **kwargs):
    import msgpack, msgpack_numpy as m
    file = str(file)
    if kwargs.pop('allow_np', True):
        kwargs.setdefault('default', m.encode)
    kwargs.setdefault('use_bin_type', True)
    with open(file, 'wb') as fp:
        msgpack.pack(obj, fp, **kwargs)


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


def df_dump(df, path, name='df'):
    print('dump ', path)
    df.to_hdf(path, name, mode='w')


def df_load(file, name='df', ):
    import pandas as pd
    try: 
        df = pd.read_hdf(file, name)
    except Exception as e: 
        print(e) 
        # HDF5ExtError: HDF5 error back trace
        df = pd.read_hdf(file, name)
    return df


def to_onehot(arr, mx):
    arr = np.asarray(arr, dtype=int) 
    assert len(arr.shape) == 1
    nelems = arr.shape[0]
    res = np.zeros((mx, nelems), dtype=int)
    res[arr, np.arange(nelems)] = 1
    return res


def to_categorical(arr):
    return np.argmax(arr, axis=0)


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


def filter_large_graph(ds, min_num=1, max_num=np.inf, mode='edge'):
    assert mode in ['edge', 'node']
    info = ds.cache_size_info()
    info_per_graph = info[f'n{mode}s']
    idxs = list(info.loc[(info_per_graph < max_num) & (info_per_graph >= min_num),
                         'idx'])
    print(f'original {len(ds)} graphs, filtered, now {len(idxs)} graphs')
    ds = copy.copy(ds)
    ds._indices = idxs
    return ds

@torch.no_grad() 
def labels_to_balanced_weights(labels, merge_lu=True): 
    ## merge_lu for two-sides 
    res=torch.zeros(3).to(labels.device)
    lbl,cnt=torch.unique(labels, return_counts=True) 
    wei=cnt.sum()/cnt
    if len(lbl)==2: 
        ## one-side 
        res[lbl]=wei 
    else:
        ## two-side  
        res[lbl]=wei 
        if merge_lu: res[0]=res[2]=(res[0]+res[2])/2.
    return res 

def extract_fn(inp, suf=None):
    res=''
    know_sufs = ['mps', 'gz', 'bas', 'tar', 'pk','log','sol'
                'txt', 'json', 'sort' 
                ]
    for r in osp.basename(inp).split('.'):
        if r not in know_sufs :  
            res+=r+'.' 
    return res[:-1] 


def stat(arr):
    if type(arr).__module__ == 'torch':
        arr = (arr).detach().cpu().numpy()
    array = np.asarray(arr)
    array = array[~np.isnan(array)]
    return dict(zip(
        ['min', 'mean', 'median', 'max', 'shape', 'std'],
        [np.min(array), np.mean(array), np.median(array), np.max(array), np.shape(array), np.std(array)]
    ))


def sparse_mat_div_by_vec(A, vec, axis='row'):
    vec = (vec).flatten()
    if axis == 'row':
        B = A.tocsr(copy=True)
        B.data /= np.repeat(vec, B.getnnz(axis=1))
        return B
    else:
        B = A.tocsc(copy=True)
        B.data /= np.repeat(vec, B.getnnz(axis=0))
        return B


def count_nonzero_sparse_mat(A, by='col'):
    nrows, ncols = A.shape
    row, col = A.nonzero()
    if by == 'col':
        # coloumn-wise: nnz(A[:,i]) for each col i
        nnz = np.zeros(ncols)
        colidx, cnts = np.unique(col, return_counts=True)
        nnz[colidx] = cnts
    else:
        nnz = np.zeros(nrows)
        rowidx, cnts = np.unique(row, return_counts=True)
        nnz[rowidx] = cnts
    return nnz


def cos_sim_vec_and_sparse_mat(v, A, bound=1e8):
    assert isinstance(v, np.ndarray)
    v = np.clip(v, -bound, bound)
    nrm_b = np.sqrt((v ** 2).sum())
    nrm_cols = np.sqrt((A.multiply(A)).sum(0))
    nrm_cols = np.array(nrm_cols).flatten()
    dot = (v * A)
    assert (dot[nrm_cols == 0] == 0).all()
    nrm_cols[nrm_cols == 0] = 1e-6
    if nrm_b == 0: nrm_b = 1e-6
    cos_sim = dot / (nrm_b * nrm_cols)
    return cos_sim


def cos_sim_sparse_mat_and_vec(A, l, bound=1e8):
    return cos_sim_vec_and_sparse_mat(l, A.T, bound)


def expand_inf(l):
    ori = l.copy()
    tag = np.zeros_like(l)
    tag[ori == np.inf] = 1
    tag[ori == -np.inf] = -1
    ori[np.abs(ori) == np.inf] = 0
    return np.stack((ori, tag), axis=1)


def concatenate_on_lst_dim(*args):
    res = []
    for a in args:
        if len(a.shape) == 1:
            a = a.reshape(-1, 1)
        res.append(a)
    return np.concatenate(res, axis=1)


class Singleton():
    def __copy__(self):
        return self

    def __deepcopy__(self):
        return self


''' Plotting classes - either tensorboard, display or save the file'''
class Tensorboard(Singleton):
    # separate for each model trained
    def __init__(self, path_log_dir):
        import os
        from torch.utils.tensorboard.writer import SummaryWriter
        self.path_tb_sw_dir = os.path.join(path_log_dir, 'tb_logs')
        self.writer = SummaryWriter(self.path_tb_sw_dir)
        self.add_scalar = self.writer.add_scalar
        self.add_scalars = self.writer.add_scalars
        self.add_histogram = self.writer.add_histogram
        self.add_image = self.writer.add_image
        self.add_images = self.writer.add_images
        self.add_figure = self.writer.add_figure
        self.add_text = self.writer.add_text
        self.add_graph = self.writer.add_graph
        self.add_embedding = self.writer.add_embedding
        self.add_pr_curve = self.writer.add_pr_curve
        self.add_custom_scalars = self.writer.add_custom_scalars
        self.add_hparams = self.writer.add_hparams
        # self.flush = self.writer.flush
        self.close = self.writer.close


class Logger(Singleton):
    def __init__(self, fpath=None, console=sys.stdout, mode='a'):
        self.console = console
        self.file = None
        if fpath is not None:
            shell(f'mkdir -p {os.path.dirname(fpath)}')
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_file_logger_prt(path):
    path = str(path) + '/'
    sys.stdout = Logger(path + 'log-prt')
    sys.stderr = Logger(path + 'log-prt-err')



class printBlocker:
    def __init__(self) -> None:
        self.backup=None 

    def blk(self):
        self.backup = sys.stdout 
        sys.stdout = open(os.devnull, 'w')
        # sys.stderr = open(os.devnull, 'w')
    
    def unblk(self):
        if self.backup is not None: 
            sys.stdout = self.backup 
        else: 
            sys.stdout = sys.__stdout__ 



def set_stream_logger(log_level=logging.INFO):
    import colorlog
    sh = colorlog.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(
        colorlog.ColoredFormatter(
            ' %(asctime)s %(filename)s [line:%(lineno)d] %(log_color)s%(levelname)s%(reset)s %(message)s',
            no_color=True,
        ))
    logging.root.addHandler(sh)
    logging.info('stream logging set up ')
    return sh


def set_file_logger(work_dir='/tmp', log_level=logging.INFO):
    mkdir_p(work_dir)
    fh = logging.FileHandler(os.path.join(work_dir, 'log-ing'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'))
    logging.root.addHandler(fh)
    return fh


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        logging.info(f'Function {func.__name__!r} executed in {(t2 - t1):.1f}s')
        return result

    return wrap_func


def init_seeds(seed=0, deterministic=False):
    import random
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    # if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
    #     torch.use_deterministic_algorithms(True)
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    from pathlib import Path
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return str(path) + '/'


def cvt_str2lst(now, splt='_', func=int):
    if isinstance(now, (list, tuple)): return now
    t = now.strip(splt).split(splt)
    t = [func(tt) for tt in t]
    return t


def cvt_lst2str(now, splt='_', ):
    if isinstance(now, str): return now
    now = [str(n) for n in now]
    return splt.join(now)

def extract_nrepair(out): 
    import re
    if out.endswith('.log'):
        assert osp.exists(out), out
        with open(out, 'r') as f:
            out_lines = f.read().split('\n')  # f.readlines()
    else:
        out_lines = out.split('\n') 
    
    for line in out_lines:
        if 'CPLEX' in line: whose_log = 'cplex' 
        if 'HiGHS' in line: whose_log = 'highs' 

    num_basis, n_slack_basis_begin, n_add_slack = -1,-1,-1 
    if whose_log =='highs': 
        n_add_slack=0
        pred_basis = f'{osp.dirname(out)}/../../pred-basis/{extract_fn(out)}.bas' 
        from scripts.cvt_to_pkl import read_bas_highs 
        cs,vs=read_bas_highs(pred_basis)
        num_basis = len(cs) 
        n_slack_basis_begin = (cs==1).sum() 
        for line in out_lines: 
            mtch = re.findall(r'Rank_deficiency (\d+)', line)
            if len(mtch)>0: 
                n_add_slack = int(mtch[0]) 
        
    return num_basis, n_slack_basis_begin, n_add_slack

def extract_fact_time(out): 
    import re
    if out.endswith('.log'):
        assert osp.exists(out), out
        with open(out, 'r') as f:
            out_lines = f.read().split('\n')  # f.readlines()
    else:
        out_lines = out.split('\n') 
    
    for line in out_lines:
        if 'CPLEX' in line: whose_log = 'cplex' 
        if 'HiGHS' in line: whose_log = 'highs' 
    
    if whose_log=='highs': 
        for line in out_lines: 
            mtch = re.findall(r'time elapsed for factorize: ([e\-\d\.]+)',line) 
            if len(mtch)>0: 
                # logging.info(f'{line} {mtch} {out}')
                return float(mtch[0] )

    return -1. 

def extract_time(out):
    ''' error code: 
    -1 -- optimal not found 
    -2 -- unknown error, may be core dump 
    -3 -- no file exists 
    '''
    import re
    if out.endswith('.log'):
        assert osp.exists(out), out
        with open(out, 'r') as f:
            out_lines = f.read().split('\n')  # f.readlines()
    else:
        out_lines = out.split('\n')
    whose_log = 'cplex' 
    iters, tm = 0, None # because highs give no output if iters=0
    # currently read first one, may start from last one
    for line in out_lines:
        if 'CPLEX' in line: whose_log = 'cplex' 
        if 'HiGHS' in line: whose_log = 'highs' 
        
        if whose_log=='cplex' and line[:13] == 'Solution time':
            line_split = [x for x in line.split(' ') if x != '']
            iters = int(line_split[7])
            tm = float(line_split[3])
            return iters, tm
        
        if 'unable open file' in line: 
            print('unable open ERROR ', out.split()[0])
            return -3,-3 

        if whose_log=='highs' and 'status' in line and 'Model' in line: 
            if not 'Optimal' in line: return -1,-1 
        if whose_log=='highs' and 'iterations' in line: 
            iters = int(re.findall(r'\d+', line)[0])
        if whose_log=='highs' and 'run time' in line: 
            tm=float(re.findall(r'\d*\.\d*', line)[0]) 
            return iters, tm 

    logging.error(f'parse fail {out}')
    return -2,-2 

def split_out_chunk(fns, chunk=None): 
    if chunk is not None and chunk.lower()!='none':  
        nfns=len(fns) 
        ck, ttl = map(float, chunk.split('/') )
        sta, ed = int(nfns/ttl*ck), int(nfns/ttl*(ck+1)) 
        logging.info(f'chunk: {nfns} {sta} {ed}')
        fns = fns[sta:ed] 
    return fns 

def get_now_time(): 
    from datetime import datetime
    now = datetime.now() # current date and time 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    # print("date and time:",date_time)
    return date_time 

def check_df(df,val): 
    row,col=np.where(df==val) 
    row=np.unique(row)
    col=np.unique(col)
    return df.iloc[row,col]

class Environment():
    def __init__(self, **kwargs):
        import argparse
        logging.root.handlers = []
        set_stream_logger()
        self.timer = Timer()
        self.osenv = dict(os.environ)
        self.show_resource()
        ''' Setting up libraries'''
        self.__setup_libraries()

        ''' project level arguments'''
        self.root_path = os.path.abspath(os.path.dirname(__file__))
        self.home_path = os.path.expanduser('~')
        os.chdir(self.root_path)
        shell('pwd')       
        self.prefetch_factor = 4

        ''' setup parser and add default arguments'''
        self.parser = argparse.ArgumentParser(
            description=f'Project: ', conflict_handler='resolve')
        self.__add_default_arguments(kwargs)
        self.parse_arguments()
        self.postprocess_args()
        print('proced args', self)
        import atexit
        atexit.register(self.exit_handler)
        
        logging.root.handlers = []
        set_stream_logger()
        set_file_logger(self.log_dir)

    def __setup_libraries(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)
        pd.set_option('display.precision', 3)
        np.set_printoptions(precision=3)

    def show_resource(self):
        if getattr(self, 'verbose', 0):
            shell('nvidia-smi', verbose=True)
            # shell('df -h')
            shell('free -h',verbose=True)

    def exit_handler(self):
        self.timer.since_start('Script ended successfully. Total time taken:')
        print('-' * 60)

    def __add_default_arguments(self, kwargs=None):
        def normlz_solver_prefidx(sv:str): 
            if not sv.endswith('-'): sv+='-' 
            assert sv in ['highs-'], sv
            return sv 
        def str_to_float(s):
            if s=='None': return None 
            else: return float(s) 
        parser = self.parser
        parser.add_argument('--dev', type=int, default=0)
        parser.add_argument('--exp_nm', type=str, default='tmp')
        parser.add_argument('--opt', type=str, default='adam')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--arch', type=str, default='GCN_FC(8,8,hids=128)', )
        parser.add_argument('--seed', type=int, default=0) 
        parser.add_argument('--num_workers', type=int)
        parser.add_argument('--batch_size', type=int, )
        parser.add_argument('--load_from', type=str, default='None')
        parser.add_argument('--dataset', type=str, default='None') 
        parser.add_argument('--solver_prefix', type=normlz_solver_prefidx, default='highs-')
        parser.add_argument('--data_prefix', type=str, ) 
        parser.add_argument('--exist_ok', type=int, default=1)
        parser.add_argument('--edge_num_thresh', type=float, )
        parser.add_argument('--verbose', type=int, default=0)
        parser.add_argument('--skip_exist', type=int, default=0) 
        parser.add_argument('-f', type=str, )  
        parser.add_argument('--init_method', type=str) 
        parser.add_argument('--chunk', type=str, default=None) 
        parser.add_argument('--loss', type=str, default='balanced')  
        parser.add_argument('--inference_manager', type=str, default="InferenceManager(0,)")
        parser.add_argument('--lp_method', type=str, default='1') # default dual 
        parser.add_argument('--gW', type=str_to_float, default=None)
        # the following may not need to be exposed 
        parser.add_argument('--split', type=str, default='val') # split only support val trainval now 
        parser.add_argument('--log_prefix', type=str, )  
        parser.add_argument('--fp16', type=int, default=0)
        self.preprecess_args(kwargs)

    def parse_arguments(self):
        known_args, unknown_args = self.parser.parse_known_args()
        if len(unknown_args)!=0: 
            logging.error(f'{unknown_args}')
        self.ori_args=vars(known_args)
        for key, value in vars(known_args).items():
            setattr(self, key, value)

        ''' destroy this object after copying data '''
        del self.parser

    def clear_terminal(self):
        if 'TERM' in self.osenv and osp.exists(f'{self.home_path}/work/cache/'):
            shell_sp('clear') 
            

    def double_check(self, key='all'):
        if self.solver_prefix =='highs-': 
            assert self.lp_method in [1,4] 
        if key=='lp_method': return 
        assert clean_str(self.arch) in self.exp_nm 
        assert clean_str(self.dataset) in self.exp_nm 
        
    def cvt_to_str(self): 
        # self.double_check() 
        res="" 
        for k,v in self.ori_args.items(): 
            if k=='f' or k=='init_method': continue 
            res+= f'--{k} {v} ' 
        return res 

    def preprecess_args(self, kwargs_in=None):
        kwargs = {}
        
        kwargs['batch_size'] = 10240 * 8 * 4
        kwargs['edge_num_thresh'] = 4e6*3
        import multiprocessing as mp
        kwargs['num_workers'] = kwargs.get('num_workers', mp.cpu_count())        
        kwargs['dev'] = 0
        kwargs['verbose'] = 1
        hm = self.home_path
        pf = osp.abspath('./')
        kwargs['data_prefix']=f'{pf}/lp-dataset/'
        kwargs['log_prefix']=f'{pf}/runs/' 
        
        if kwargs_in: kwargs.update(kwargs_in)  
        if kwargs: self.parser.set_defaults(**kwargs)

    def get_method_sfx(self, ):
        self.postprocess_lp_method() 
        if int(self.lp_method)==1: 
           return '' 
        return f'-m{self.lp_method}'  

    def postprocess_lp_method(self, ): 
        assert self.solver_prefix=='highs-', self.solver_prefix
        lp_method_map = dict(dual=1, primal=4)  
        if self.lp_method in lp_method_map: 
            self.lp_method = lp_method_map[self.lp_method] 
        self.lp_method = int(self.lp_method) 

    def postprocess_args(self, ):
        self.dataset_prefix = f'{self.data_prefix}/{self.dataset}'
        self.dataset_processed_prefix = f'{self.dataset_prefix}/{self.solver_prefix}inp_tgt{self.get_method_sfx()}/'
        self.log_dir=f'{self.log_prefix}/{self.exp_nm}'         
        init_seeds(self.seed)
        self.log_dir = increment_path(self.log_dir, exist_ok=self.exist_ok) 
        self.dev = f'cuda:{self.dev}' if torch.cuda.is_available() else 'cpu'
        self.postprocess_lp_method() 
 
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        output = ''
        for key, value in vars(self).items():
            if key == 'osenv': continue
            output += f'{key} : {value} \n'
        return output


def clean_str(exp_nm):
    for i in '()=,.': exp_nm=exp_nm.replace(i, '-') 
    return exp_nm

def parse_str(inp, key): 
    if key=='dataset': 
        tries=['small-perm', 'medium-inv', 'medium-perm', 'medium', 'mirp', 'small', 'libsvm_6', 'miplib_8', 
                'generated_15-10', 'generated_15-1', 
                'stoch-sc-5','stoch-sc', 'stoch-el',] 
        for t in tries: 
            if t in inp: return t 
    elif key=='arch': 
        for hids in [128,1024]:
            for depth in [3,5,7,9]: 
                t=f'GCN_FC(8,8,hids={hids},depth={depth})' 
                if clean_str(t) in inp: return t 
    elif key=='solver_prefix':
        for t in ['highs-']: 
            if t in inp: return t 
        return '' 
    
    else: raise ValueError('n imp')
    raise ValueError('parse fail')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        import collections
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.mem = collections.deque(maxlen=10) # 10  

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # way 1
        val = float(val)
        self.mem.append(val)
        self.avg = np.mean(list(self.mem))
        ## way 2
        # self.val = val
        # self.sum += val * n
        # self.count += n
        # self.avg = self.sum / self.count

def batch_to(batch, dev, half=False):
    nms=['edge_index', 'x_s', 'x_t', 'y_s','y_t'] 
    for nm in nms: 
        if half: 
            batch[nm]=batch[nm].half() 
        batch[nm]=batch[nm].to(dev) 
    return batch 
    
def proc_iters(s):
    func = cvt_lrg_int_to_latex
    mn = s["mean"] 
    std=s["std"] 
    res = f'${func(mn)}' 
    if np.abs(std)>1e-5: res+='{\scriptscriptstyle ' + f' \pm {func(std)}' +'}'
    res+='$'
    return res 
def proc_percent(s):
    s=s*100 
    std=s["std"]
    res = (f'${s["mean"]:.1f} ' ) 
    if np.abs(std)>1e-5: res+= ('{\scriptscriptstyle ' 
        f'\pm {cvt_flt_to_latex(s["std"])}' '}' ) 
    res+='$'
    return res
def proc_flt(s, ): 
    func=cvt_flt_to_latex
    mn=s["mean"] 
    std=s["std"] 
    res = f'${func(mn)}' 
    if np.abs(std)>1e-5: res+='{\scriptscriptstyle ' + f'\pm {func(std)}' +'}' 
    res+='$'
    return res 
def cvt_lrg_int_to_latex(x): 
    if x>=1e6:
        reduce,reducer=int(1e6),'M' 
    elif x>=1e3: 
        reduce,reducer=int(1e3),'K'    
    else: 
        reduce,reducer=1, ''
    x/=reduce
    return f'{x:.1f}{reducer}' 
def cvt_flt_to_latex(x): 
    if float(x)>=0.05: 
        x=f'{x:.1f}' 
    else: 
        x=f'{x:.0e}' 
        x=x.replace('-0','-')
        x= x.replace('e-', '\text{e-}')
    return x 

def proc(s):
    nm=(s.name) 
    for sub_nm in ['acc', 'prec', 'recl', 'repair', 'impr']:
        if sub_nm in nm: return proc_percent(s)  
    for sub_nm in ['inf_time', 'fact_time']: 
        if sub_nm in nm: return proc_flt(s)  
    return proc_iters(s) 

def filter_cols(c1,c2): 
    cols=[] 
    for c in c1:
        if c in c2: cols.append(c)  
    return cols 
