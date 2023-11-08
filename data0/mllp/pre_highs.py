import sys, os, time, logging, glob
import scipy
import numpy as np
import torch
import time as tm
import re

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

    
def extract_iter(text):
    pattern = r'Simplex   iterations:(.*)'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
    else:
        content = ""
    
    return content

def extract_time(text):
    pattern = r'HiGHS run time      :(.*)'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
    else:
        content = ""
    
    return content

def extract_value(text):
    pattern = r'Objective value     :(.*)'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
    else:
        content = ""
    
    return content




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





import click
  
@click.command()
@click.option('--name', prompt='Your name', help='The person to greet.')
@click.option('--model_name', prompt='Your name', help='The person to greet.')

def run(name, model_name):
    start = tm.time()
    method = "dual"

    if method == "dual":#dont touch!!!!!
        md = 1
    else:
        md = 4

    time_limit = 10000

    #path = "processed/"+name+"_eval"
    #dest_path = "highs/mps0000s"
    path = "../sib/lp-dataset/"+name+"ppp/mps"
    ppath = "../sib/lp-dataset/"+name+"ppp/highs-inp_tgt/raw"
    highs_path = "highs/eval_"+name+"_"+model_name
    result_path = "pred_res/eval_"+name+"_"+model_name ###########-----------
    os.makedirs(result_path, exist_ok = True)
    check_list = []
    files = os.listdir(ppath)
    #files = os.listdir(path)
    file_num = 0
    files = list(files)
    files = sorted(files, key=lambda nm: (len(nm), nm))
    files = np.array(files)
    print(files)
    train_files, val_files = split_train_val(files, 0)
    files = val_files

    iter = []
    time = []
    value = []
    #iter_ratio = 0
    #time_ratio = 0
    num = 0
    num0 = 0

    for file in files:
        num0 += 1
        print("================ instance {} =================".format(num0))
        #if file != 'maros-r7.mps':
            #continue
        print(file)
        file = file.split('.')[0]+".mps"
        print(file)
        if (file == "prob_2.mps"):
            continue


        MPS = path+"/"+file
        BAS0 = highs_path+"/"+file.split('.')[0]+".pk.bas"
        #BAS = result_path+"/"+file+".bas"
        print(MPS)
        #cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 --time_limit {time_limit} -ss {md}  ' 
        cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 --time_limit {time_limit} -bi {BAS0} -ss {md}  ' 
        out, _ = shell(cmd)
        print(out)
        iter1 = extract_iter(out)
        time1 = extract_time(out)
        value1 = extract_value(out)
        if iter1 != '' and time1 != '' and value1 != '':
            iter1 = int(iter1)
            time1 = float(time1)
            value1 = float(value1)
        else:
            iter1 = -1
            time1 = -1
            value1 = -1
        '''cmd = f'highs --model_file {MPS} --presolve off --solver simplex --random_seed 0 --time_limit 600 -bi {BAS0} -ss {1}  ' 
        out, _ = shell(cmd) 
        print(out)
        iter2 = extract_iter(out)
        time2 = extract_time(out)
        if iter2 != '' and time2 != '':
            iter2 = int(iter2)
            time2 = float(time2)
        else:
            continue
        if time1 != 0:
            time_ratio += time2 / time1
        else:
            continue
        iter_ratio += iter2 / iter1
        iter.append([iter1, iter2])
        time.append([time1, time2])'''
        iter.append(iter1)
        time.append(time1)
        value.append(value1)
        if time1 >= 10000:
            num += 1

    #iter_ratio /= num
    #time_ratio /= num
    print(iter)
    print(time)
    print(value)
    np.save(result_path+"_iter.npy", np.array(iter))
    np.save(result_path+"_time.npy", np.array(time))
    np.save(result_path+"_value.npy", np.array(value))

    #print(iter_ratio)
    #print(time_ratio)
    print("----------------")
    print(num)
    print(num0)
    print(tm.time()-start)

if __name__ == '__main__':
    run()