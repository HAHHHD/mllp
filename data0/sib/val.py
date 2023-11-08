import logging
import utils
from utils import *
import torch.nn
from dataset import LPDataset, UnipartiteData, MyToBipartite
from arch import *
from torch_geometric.loader import DataLoader, DynamicBatchSampler, NeighborLoader
from tqdm import tqdm
from scripts.cvt_to_pkl import read_bas
dev='cpu'
if torch.cuda.is_available(): dev='cuda:0'
@torch.no_grad() 
def model_inference_with_batch(model, batched_graphs, args=None):   
    depth = re.findall(r'depth=(\d+)', args.arch) 
    if len(depth)==0: depth = 2
    else: depth = int(depth[0]) - 1
    batch_size=args.batch_size
    batch_size=min(batch_size, batched_graphs.num_nodes) 
    if hasattr(batched_graphs, 'x_s'): # already is bipartite 
                loader = [batched_graphs]
    else:
        loader = NeighborLoader(batched_graphs, input_nodes=None, directed=False,
                            num_neighbors=[-1]*depth, shuffle=False,
                            transform=MyToBipartite(thresh_num=np.inf), batch_size=batch_size,
                            num_workers=8, pin_memory=False,
                            drop_last=False, prefetch_factor=5
                            ) 
    model.eval() 
    lc,lv=[],[] 
    for batch in tqdm(loader): 
        batch_to(batch, dev, args.fp16) # only one gpu 
        logit_cons, logit_vars = model(batch)
        logit_cons, logit_vars = logit_cons[:batch.s_bs], logit_vars[:batch.t_bs]
        lc.append(logit_cons.cpu()); lv.append(logit_vars.cpu()) 
    lc,lv=torch.cat(lc, dim=0), torch.cat(lv, dim=0) 
    return lc, lv
    # MyToBipartite(thresh_num=np.inf)(batched_graphs)
    # lc2,lv2=model(batched_graphs.cuda())
    # lc2,lv2=lc2.cpu(), lv2.cpu() 
    # assert (lc-lc2).abs().sum().item() < 1 
    # assert (lv-lv2).abs().sum().item() < 1 

@torch.no_grad()
def validation(model, loader, dev, dump_info=None):
    if dump_info and osp.exists(dump_info): df = df_load(dump_info) 
    else: df = None 
    model.to(dev)
    model_ori_training = model.training 
    model.eval()
    avg_acc = avg_loss= 0. 
    for idx, batch in tqdm(enumerate(loader)):
        fn = extract_fn(batch.processed_path[0])
        logit_cons, logit_vars = model_inference_with_batch(model, batch, args) # todo this args is global vars
        batch = MyToBipartite(thresh_num=np.inf)(batch)
        acc, prec, recl = accuracy(
            torch.cat((logit_cons, logit_vars), dim=0),
            torch.cat((batch.y_s, batch.y_t), dim=0), logit_cons.shape[0], 
            return_pr=True
        )
        avg_acc += acc / len(loader) 
        if df is not None: 
            df.loc[df.fn == fn, 'acc'] = acc 
            df.loc[df.fn == fn, 'prec'] = prec 
            df.loc[df.fn == fn, 'recl'] = recl 
        if idx % 9 == 1: print('\n', idx, fn, len(loader), acc, prec, recl)
    if model_ori_training: model.train()
    if df is not None: df_dump(df, f'{args.log_dir}/time.h5')
    return avg_loss, avg_acc

@torch.no_grad()
def validation_wrt_converged(model, loader, dev, dump_info=None):
    if dump_info and osp.exists(dump_info): df = df_load(dump_info) 
    else: df = None 
    model_ori_training = model.training 
    model.to(dev)
    model.eval()
    avg_acc = 0. 
    for idx, batch in tqdm(enumerate(loader)):
        con_nms = batch.con_nms[0]
        var_nms = batch.var_nms[0]
        fn = extract_fn(batch.processed_path[0])
        logit_cons, logit_vars = model_inference_with_batch(model, batch, args)
        tgt= f'{args.log_dir}/opt-from-pred-basis/{fn}.bas' 
        if osp.exists(tgt): 
            con_lbls, var_lbls = read_bas(
                # f'{args.dataset_prefix}/{args.solver_prefix}basis/{fn}.bas', 
                tgt, 
                con_nms, var_nms 
            ) 
            con_lbls, var_lbls = torch.from_numpy(con_lbls), torch.from_numpy(var_lbls)
            acc, prec, recl = accuracy(
                torch.cat((logit_cons, logit_vars), dim=0),
                torch.cat((con_lbls, var_lbls), dim=0), logit_cons.shape[0], 
                return_pr=True
            )
            avg_acc += acc / len(loader) 
            if df is not None: 
                df.loc[df.fn == fn, 'cvg/acc'] = acc 
                df.loc[df.fn == fn, 'cvg/prec'] = prec 
                df.loc[df.fn == fn, 'cvg/recl'] = recl 
            if idx % 9 == 1: print('\n', idx, fn, len(loader), acc, prec, recl)
    if model_ori_training: model.train()
    if df is not None: df_dump(df, f'{args.log_dir}/time.h5')
    return 0, avg_acc

@torch.no_grad()
def inference_gnn(logits, m, **kwargs): 
    n = logits.shape[0] - m
    pr = F.softmax(logits.float(), dim=-1) 
    pr = pr.clone()
    pr[torch.isnan(pr)]=0 # if half model, nan will occur 

    _, topk_idx = pr[:, 1].topk(m)
    pr[:, 1] = pr.min() - 1
    pr[topk_idx, 1] = pr.max() + 1
    pred = pr.argmax(-1)

    ## the total number of basic variables—both structural and row—is equal to the number of rows in the constraint matrix
    assert (pred == 1).sum().item() == m
    ## the number of basic structural variables is equal to the number of nonbasic row variables
    assert (pred[m:m + n] == 1).sum().item() == \
           ((pred[:m] == 0) | (pred[:m] == 2)).sum().item()

    return pred

# deprecated: inference manager, the following for sparsity+GNN 
@torch.no_grad() 
def inference_all_slacks(logits, m, **kwargs): 
    n = logits.shape[0] - m
    pr = F.softmax(logits, dim=-1) 
    pr = pr.clone()
    pr[:, 1] = pr.min() - 1
    pred = torch.ones((m+n), dtype=torch.long)  
    pred[m:]=pr[m:,:].argmax(-1) 
    return pred 

@torch.no_grad() 
def inference_gnn_sparsity(logits, m, nnzs, mode='add', gnn_wei=.5):
    assert mode is not None 
    n = logits.shape[0] - m
    pr = F.softmax(logits, dim=-1) 
    pr = pr.clone()

    nnzs[nnzs==0]=nnzs.max()+1 # the variables that do not appears in A should be non-basic
    nnzs=1./nnzs
    # nnzs-=nnzs.mean() 
    # pr_nnzs=torch.sigmoid(nnzs) 
    nnzs /= nnzs.sum() 
    nnzs *= m  
    pr_nnzs=nnzs 
    pr_gnn = pr[:,1] 
    if mode=='add': 
        assert gnn_wei is not None 
        pr_basis = gnn_wei * pr_gnn + (1-gnn_wei) * pr_nnzs 
    else: 
        assert mode == 'mult'
        pr_basis = pr_gnn * pr_nnzs 
    pr[:,1]=pr_basis 

    _, topk_idx = pr[:, 1].topk(m)
    pr[:, 1] = pr.min() - 1
    pr[topk_idx, 1] = pr.max() + 1
    pred = pr.argmax(-1)

    return pred 

class InferenceManager:
    '''
    --inference_manager "InferenceManager(0, run=0)"
    InferenceManager(1, run=0) 
    InferenceManager(2,1) # mult 
    InferenceManager(2,0,.1) # add 
    InferenceManager(2,0,.9) 
    '''
    def __init__(self, which_func, mode=None, gnn_wei=None, run=0): 
        mp=['inference_gnn', 'inference_all_slacks', 'inference_gnn_sparsity']
        self.which_func=mp[which_func]
        if mode is not None and isinstance(mode, int): mode=['add','mult'][mode] 
        self.mode=mode 
        self.gnn_wei=gnn_wei  
        self.run=run 

    def get_log_folder(self): 
        if self.which_func == 'inference_gnn': 
            return f'gnn-bas-{self.run}'
        else:
            return self.get_basis_folder() 

    def get_basis_folder(self): 
        if self.which_func == 'inference_gnn': 
            res=f'pred-basis' 
            if self.run!=0: res+=f'-{self.run}'
            return res 
        elif self.which_func == 'inference_all_slacks': 
            return f'all-slacks-bas-{self.run}' #
        elif self.which_func == 'inference_gnn_sparsity': 
            return f'gnn-sparsity-{self.mode}-{self.gnn_wei}-{self.run}'  
#--------------------------------------------------------------------------------
@torch.no_grad()
def basis_accuracy(logits, gt, num_cons):
    #-----------------------
    #pred = inference_gnn(logits, num_cons) 
    num_cons = np.sum(np.sum(gt.cpu().numpy() == 1))
    pred_indices = torch.topk(logits, k=num_cons)[-1].cpu().detach().numpy()
    pred = np.zeros(logits.shape[0])
    pred[pred_indices] = 1
    #if torch.unique(pred[:num_cons]).shape[0] == 1 and torch.unique(pred[:num_cons]).item()==1:
        #logging.warning('warning: may collapse, basis==all slacks') 
    
    # acc = (pred.cpu() == gt).float().mean().item() 
    pred_np, gt_np = pred, gt.cpu().numpy() 
    #pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy() 

    pred_basis = (pred_np == 1)
    gt_basis = (gt_np == 1)
    basis_acc = np.logical_and(pred_basis, gt_basis).sum() / num_cons
    # acc = (gt_np==pred_np).mean() 
    ## because pred top-m knowledges is added, so prec == recl in following  
    # prec = metrics.precision_score(gt_np, pred_np, labels=[1], average='macro')
    # recl = metrics.recall_score(gt_np, pred_np, labels=[1], average='macro') 


    return basis_acc
#--------------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits, gt, num_cons, return_pr=False, dataset_name=''):
    from sklearn import metrics 
    pred = inference_gnn(logits, num_cons)    
    if torch.unique(pred[:num_cons]).shape[0] == 1 and torch.unique(pred[:num_cons]).item()==1:
        logging.warning('warning: may collapse, basis==all slacks') 
    
    # acc = (pred.cpu() == gt).float().mean().item() 
    pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy() 

    acc1 = (gt_np[:num_cons]==pred_np[:num_cons]).mean()
    acc2 = (gt_np[num_cons:]==pred_np[num_cons:]).mean() 
    if dataset_name and 'stoch' in dataset_name: 
        ## because stoch constraints is always labeled as non-basic 
        acc1=acc2 
    acc = (acc1+acc2) / 2. 
    # acc = (gt_np==pred_np).mean() 
    ## because pred top-m knowledges is added, so prec == recl in following  
    # prec = metrics.precision_score(gt_np, pred_np, labels=[1], average='macro')
    # recl = metrics.recall_score(gt_np, pred_np, labels=[1], average='macro') 

    metric_kwargs=dict(labels=[1], average='macro') ## becaseu all mps are one-side inequalities 
    p1=metrics.precision_score(gt_np[:num_cons], pred_np[:num_cons],**metric_kwargs) 
    p2=metrics.precision_score(gt_np[num_cons:], pred_np[num_cons:],**metric_kwargs) 

    r1=metrics.recall_score(gt_np[:num_cons], pred_np[:num_cons],**metric_kwargs) 
    r2=metrics.recall_score(gt_np[num_cons:], pred_np[num_cons:],**metric_kwargs) 
    if dataset_name and 'stoch' in dataset_name: 
        ## because stoch constraints is always labeled as non-basic 
        p1=p2; r1=r2
    prec = (p1+p2)/2. 
    recl = (r1+r2)/2. 

    # metrics.confusion_matrix(gt_np[:num_cons], pred_np[:num_cons], )
    # metrics.confusion_matrix(gt_np[num_cons:], pred_np[num_cons:], )
    if return_pr: 
        return acc, prec, recl, #prec2, recl2 
    else:
        return acc


if __name__ == '__main__':
    args = utils.Environment(
    )

    log_dir = args.log_dir
    set_file_logger_prt(args.log_dir)

    dev = args.dev
    ds = LPDataset(args.dataset_processed_prefix,
                   chunk=None, transform=MyToBipartite(thresh_num=args.edge_num_thresh),                    
                   )
    assert len(ds)!=0, 'should not empty'
    train_ds, val_ds = split_train_val(ds,args.seed)
    info=ds.cache_size_info() 
    info.loc[train_ds.indices(), 'split']='train'
    info.loc[val_ds.indices(), 'split']='val' 
    print('len val ds', len(val_ds))
    num_workers = args.num_workers
    kwargs = dict(batch_size=1, num_workers=num_workers,shuffle=False,
                persistent_workers=num_workers > 0,
                drop_last=False, prefetch_factor=args.prefetch_factor, 
                pin_memory=not train_ds.load_meta ) 
    val_ds.load_meta = True 
    val_loader = DataLoader(val_ds, **kwargs)
    train_loader = DataLoader(train_ds, **kwargs)

    model = eval(args.arch).to(dev)
    if args.load_from.lower() != 'none':
        model.load(args.load_from)
    if args.fp16: model.half() 
    model.eval()

    dump_info_pn = f'{args.log_dir}/time.h5' 
    if not osp.exists(dump_info_pn): df_dump(info, dump_info_pn)
    from functools import partial 
    accuracy = partial(accuracy, dataset_name = args.dataset) 

    # avg_loss, avg_acc = validation_wrt_converged(model, val_loader, dev, dump_info=dump_info_pn)
    # print('avg val wrt converged acc', avg_acc)

    avg_loss, avg_acc = validation(model, val_loader, dev, dump_info=dump_info_pn)
    print('avg val acc', avg_acc)

    # avg_loss, avg_acc = validation(model, train_loader, dev, dump_info=dump_info_pn)
    # print('avg train acc', avg_acc)





