import os, sys
sys.path.insert(0, os.path.dirname(__file__) + '/../')
import utils
from utils import *
from arch import *
from dataset import *
from val import * 
from threading import Thread
from torch_geometric.loader import DataLoader

def enc_vec(vbas):
        return ' '.join([str(v) for v in vbas])

def write_bas_highs(fn, vnms=None, cnms=None, vbas=None, cbas=None):
    assert vbas is not None   
    # logging.info(f'write to {fn}')
    mkdir_p(osp.dirname(fn))
    with open(fn,'w') as f: 
        f.write('HIGHS v1\nValid\n')  
        f.write(f'# Columns {len(vbas)}\n') 
        f.write(enc_vec(vbas)+'\n')  
        f.write(f'# Rows {len(cbas)}\n') 
        f.write(enc_vec(cbas)+'\n') 

def write_bas(fn, var_nms, con_nms, pred_var, pred_con):
    # * ENCODING=ISO-8859-1
    out = f'''NAME          0.mps  Iterations 0  Rows {len(con_nms)}  Cols {len(var_nms)} \n'''
    with open(fn, 'w') as f:
        f.write(out)

        var_nms = np.array(var_nms)
        con_nms = np.array(con_nms)
        pred_var = np.array(pred_var) 
        pred_con = np.array(pred_con) 

        var_bs = var_nms[pred_var == 1]
        con_ll = con_nms[pred_con == 0]
        con_ul = con_nms[pred_con == 2]
        assert len(var_bs) == len(con_ll) + len(con_ul)

        for v, c in zip(var_bs[:len(con_ll)], con_ll):
            f.write(f' XL {v} {c} \n')

        for v, c in zip(var_bs[len(con_ll):], con_ul):
            f.write(f' XU {v} {c} \n')
        var_ul = var_nms[pred_var == 2]
        # the following is by default, so not need to write
        # var_ll = var_nms[pred_var == 0]
        # for v in var_ll:
        #     f.write(f' LL {v} \n')

        for v in var_ul:
            f.write(f' UL {v} \n')

        f.write('ENDATA')

def write_sort_vars(fn, logits, m): 
    ## my format is m constrainst first 
    pr = F.softmax(logits, dim=-1) 
    pr_cons, pr_vars=pr[:m, :], pr[m:, :] 
    ## for  bixby crash 
    idxs_cons, idxs=pr_cons[:,1].cpu().numpy(), pr_vars[:,1].cpu().numpy()
    with open(fn, 'w') as f: 
        f.write(f'{len(idxs)} \n') # vars write first
        f.write(enc_vec(idxs) + '\n')  
        f.write(f'{len(idxs_cons)} \n')
        f.write(enc_vec(idxs_cons) + '\n')  

    
@torch.no_grad()
def proc_one_batch(batch, write_func=None, inf_mng=None):  
    # ei=batch.edge_index.cpu().clone() 
    # nedges = ei.nnz()  
    # ei = ei.set_value(torch.ones(nedges).to(ei.device()), layout='coo')
    # deg_vars = ei.sum(0)
    # nnzs=torch.cat((torch.ones(m).to(ei.device()),deg_vars)) 

    fn = extract_fn(batch.processed_path[0])+'.bas'
    # logpr_cons, logpr_vars = model(batch)   
    logpr_cons, logpr_vars= model_inference_with_batch(model, batch, args) 
    logpr_cons, logpr_vars=logpr_cons.cpu(), logpr_vars.cpu()
    m=logpr_cons.shape[0]
    #logits = torch.cat((logpr_cons, logpr_vars), dim=0).float() 
    ## pred-basis; gnn only 
    #pred = inference_gnn(logits, m, )

    y_s, y_t = batch.y_s[:batch.s_bs], batch.y_t[:batch.t_bs]
    #acc = basis_accuracy(torch.cat((logpr_cons, logpr_vars), dim=0),
                        #torch.cat((y_s, y_t), dim=0), logpr_cons.shape[0])
    acc = basis_accuracy(logpr_vars, y_t, np.sum(y_t.detach().cpu().numpy() == 1))
    ## pred-basis-all-slacks; all slacks 
    # pred = inference_all_slacks(logits, m)
    ## gnn+sparsity 
    # pred=inference_gnn_sparsity(logits, m, nnzs=nnzs) 
    
    # pred = eval(inf_mng.which_func)(logits, m, 
    #                                 nnzs=nnzs, mode=inf_mng.mode, gnn_wei=inf_mng.gnn_wei)

    '''if write_func is not None:
        con_nms = batch.con_nms[0]
        var_nms = batch.var_nms[0] 
        pred = pred.cpu().numpy()
        pred_con, pred_var = pred[:m], pred[m:]
        fn = f'{args.log_dir}/{pred_folder_nm}/{fn}'
        p = Thread(target=write_func,
                   args=(fn, var_nms, con_nms, pred_var, pred_con),
                   daemon=True 
                   ) 
        p.start()
        ps.append(p) # ps is global 
    p=Thread(target = write_sort_vars, 
            args = (fn+'.sort', logits, m), 
            daemon=True 
    ) 
    p.start() 
    ps.append(p) '''
    return acc


@torch.no_grad()
def inference_only(batch):
    num_cons = batch.x_s.shape[0]
    with torch.no_grad():
        logpr_cons, logpr_vars = model(batch)
    _ = inference_gnn(torch.cat((logpr_cons, logpr_vars), dim=0), num_cons) 


if __name__ == '__main__':
    args = utils.Environment(
    )
    inf_mng=eval(args.inference_manager)
    pred_folder_nm = inf_mng.get_basis_folder() 
    dev = args.dev
    exp_nm = args.exp_nm
    model = eval(args.arch).to(args.dev)
    model.load(args.load_from)

    shell(f'mkdir -p {args.log_dir}/{pred_folder_nm}/')
    ds = LPDataset(args.dataset_processed_prefix, 
                    MyToBipartite(thresh_num=args.edge_num_thresh),)
    num_workers = args.num_workers 
    train_ds, val_ds = split_train_val(ds,args.seed)
    ds.load_meta = train_ds.load_meta = val_ds.load_meta = True 
    # todo integrate train val ds into ds 
    loader_args = dict(batch_size=1,
                       num_workers=num_workers, shuffle=False,
                       persistent_workers=num_workers > 0,
                       drop_last=False, prefetch_factor=args.prefetch_factor, 
                       pin_memory=not train_ds.load_meta 
                       )
    train_loader = DataLoader(train_ds, **loader_args )
    val_loader = DataLoader(val_ds, **loader_args )
    if args.fp16: model.half() 
    model.eval()
    ps = []
    from itertools import chain
    write_func = write_bas_highs 
    if args.split=='val': loader=val_loader 
    else: loader=chain(val_loader, train_loader) 
    tol_acc = 0
    for idx, batch in enumerate(tqdm(loader)): 
        acc = proc_one_batch(batch, write_func=write_func, inf_mng=inf_mng)
        print(acc)
        tol_acc += acc
    tol_acc /= len(tqdm(loader))
    print(tol_acc)
    print("------------------------------------------------------------------------")
    for p in ps: p.join()

    ## bechmark time todo 
    try:
        ds = LPDataset(args.dataset_processed_prefix, 
                    MyToBipartite(thresh_num=np.inf),)
        ds.load_meta = train_ds.load_meta = val_ds.load_meta=False 
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers,
                                shuffle=True, pin_memory=True,
                                )
        dump_info_pn = f'{args.log_dir}/time.h5' 
        ds.dump_size_info(dump_info_pn) 
        df=df_load(dump_info_pn) 
        timer=Timer()  
        for idx, batch in enumerate(tqdm(val_loader)): 
            batch = batch_to(batch,dev,args.fp16) 
            timer.since_last_check('data', verbose=False)
            inference_only(batch) 
            inf_time = timer.since_last_check('inf', verbose=False)
            fn = extract_fn(batch.processed_path[0])
            df.loc[df.fn==fn, 'inf_time'] = inf_time
        df_dump(df, dump_info_pn) 
    except Exception as e: 
        print('ignore err ', e) 