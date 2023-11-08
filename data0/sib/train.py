import os
import re
import json
import numpy as np
import utils
from utils import *
import torch.nn
from dataset import LPDataset, UnipartiteData, MyToBipartite
from arch import *
from torch_geometric.loader import DataLoader, DynamicBatchSampler, NeighborLoader
from torch.optim.lr_scheduler import StepLR
from easydict import EasyDict as edict
from val import basis_accuracy, accuracy

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

cri_focal = FocalLoss() 

'''def unbalanced(logpr_cons, logpr_vars,y_s,y_t): 
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(torch.cat((logpr_cons, logpr_vars), dim=0), 
                    torch.cat((y_s,y_t), dim=0), 
                    )
    return loss '''
def unbalanced(logpr_vars,y_t): 
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logpr_vars, 
                    y_t, 
                    )
    return loss 


def balanced(logpr_cons, logpr_vars,y_s,y_t): 
    loss=0. 
    m,n = len(y_s), len(y_t)
    criterion=torch.nn.CrossEntropyLoss(weight=labels_to_balanced_weights(y_s))  
    loss+=(m+n)/m*criterion(logpr_cons, y_s) 
    criterion=torch.nn.CrossEntropyLoss(weight=labels_to_balanced_weights(y_t))   
    loss+=(m+n)/n*criterion(logpr_vars, y_t)  
    return loss 
# balance=balanced 

def focal(logpr_cons, logpr_vars,y_s,y_t): 
    loss = cri_focal(torch.cat((logpr_cons, logpr_vars), dim=0), 
                    torch.cat((y_s,y_t), dim=0),) 
    # loss = cri_focal(logpr_vars, y_t) 
    return loss 

def run_exp(args):
    dev = args.dev
    log_dir = args.log_dir
    writer = Tensorboard(log_dir)
    set_file_logger_prt(log_dir)
    try:
        writer.add_text('hyper', pd.DataFrame([args.__dict__]).T.to_markdown())
        json_dump(args.__dict__, args.log_dir + 'args.json')
    except:
        pickle_dump(args.__dict__, args.log_dir + 'args.pkl')
    ds = LPDataset(args.dataset_processed_prefix, chunk=None,
                   transform=MyToBipartite(thresh_num=args.edge_num_thresh),
                   )
    num_workers = args.num_workers 
    train_ds, val_ds = split_train_val(ds,args.seed)
    loader_args = dict(batch_size=1,
                       num_workers=num_workers, shuffle=True,
                       persistent_workers=num_workers > 0,
                       drop_last=True, prefetch_factor=args.prefetch_factor
                       )
    train_loader = DataLoader(train_ds,
                              **loader_args
                              )

    model = eval(args.arch)
    if args.load_from.lower() != 'none': model.load(args.load_from)
    # from torch_geometric.nn import DataParallel
    # model = DataParallel(model).to('cuda:0')
    #model.load(f'{log_dir}/mdl.pth')
    model = model.to(dev)

    if args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    #scheduler = StepLR(opt, step_size=max(args.epochs // 4, 1), gamma=0.1)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    glstep = 0
    # avg_loss, avg_acc = validation(model, val_loader, dev)
    # writer.add_scalar('val/loss', avg_loss, glstep)
    # writer.add_scalar('val/acc', avg_acc, glstep)
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    train_dict = {"loss":[], "basis_acc":[], "acc":[]}

    #with open("train_log_miplibCs.json",'r') as f:
        #train_dict = json.load(f)
    for epoch in range(args.epochs):
        lloss = 0
        bbasis_acc = 0
        aacc = 0
        skip = 0
        model.train()
        #if epoch % 99==1:
            #args.show_resource()
        loss_meter.reset(), acc_meter.reset()
        for idx_graphs, batched_graphs in enumerate(train_loader):
            if hasattr(batched_graphs, 'x_s'): # already is bipartite 
                loader = [batched_graphs]
            else:
                if batched_graphs.edge_index.shape[-1] == 0: continue
                batch_size = min(args.batch_size, batched_graphs.num_nodes)
                depth = re.findall(r'depth=(\d+)', args.arch)
                if len(depth)==0: depth = 2
                else: depth = int(depth[0]) - 1 # for GCN_FC, -1 because FC layer 
                loader = NeighborLoader(batched_graphs, input_nodes=None, directed=False,
                                        num_neighbors=[6]*depth, shuffle=True,
                                        transform=MyToBipartite(thresh_num=np.inf), batch_size=batch_size,
                                        num_workers=args.num_workers, pin_memory=True,
                                        drop_last=True, prefetch_factor=args.prefetch_factor
                                        )
            for batch in loader:
                batch.to(dev, non_blocking=True)
                # torch.cuda.empty_cache()
                glstep += 1
                logit_cons, logit_vars = model(batch)
                logit_cons, logit_vars = logit_cons[:batch.s_bs], logit_vars[:batch.t_bs]
                y_s, y_t = batch.y_s[:batch.s_bs], batch.y_t[:batch.t_bs]
                #--------------------------------------------------------------------
                if (sum(np.array(y_s == 3)) + sum(np.array(y_t == 3)) > 0):
                    skip += 1
                    continue
                #-----------------------------------------------------------------
                #loss = eval(args.loss)(logit_cons, logit_vars, y_s, y_t)
                #loss = eval(args.loss)(logit_vars, y_t)
                loss = criterion(logit_vars, y_t.float())

                assert not torch.isnan(loss).item()
                opt.zero_grad()             
                loss.backward()
                opt.step()

                lloss += loss.mean()
                #print(y_s.shape, y_t.shape, np.sum(y_t.detach().cpu().numpy() == 1))
                bbasis_acc += basis_accuracy(logit_vars,
                        y_t, np.sum(y_t.detach().cpu().numpy() == 1))
                '''aacc += accuracy(
                        torch.cat((logit_cons, logit_vars), dim=0),
                        torch.cat((y_s, y_t), dim=0), logit_cons.shape[0]
                    )'''
                loss_meter.update(loss.item())
                '''acc_meter.update(
                    accuracy(
                        torch.cat((logit_cons, logit_vars), dim=0),
                        torch.cat((y_s, y_t), dim=0), logit_cons.shape[0]
                    )
                )'''

                if glstep % 9 == 1:
                    logging.info(
                        f'''{epoch} {idx_graphs}/{len(train_loader)}-{skip} {glstep, len(loader),
                                                                      loss_meter.avg, acc_meter.avg}'''
                    )
                    writer.add_scalar('epoch', epoch, glstep)
                    writer.add_scalar('train/loss', loss_meter.avg, glstep)
                    writer.add_scalar('train/acc', acc_meter.avg, glstep)
                    #writer.add_scalar('train/lr', scheduler.get_last_lr()[0], glstep)
        

        train_dict["loss"].append(lloss.detach().numpy() / len(train_loader) - skip)
        train_dict["basis_acc"].append(bbasis_acc / len(train_loader) - skip)
        train_dict["acc"].append(aacc / len(train_loader) - skip)
#-------------------------------------------------------------------------------------------------------
        with open("train_log_miplibCs000.json", "w") as json_file:
            json.dump(train_dict, json_file)
        # avg_loss_val, avg_acc = validation(model, val_loader, dev)
        # writer.add_scalar('val/loss', avg_loss_val, glstep)
        # writer.add_scalar('val/acc', avg_acc, glstep)
        # print('epoch', epoch, avg_loss, avg_loss_val)
        #-------------------------------------------------------------------------
        #scheduler.step()
        model.save(f'{log_dir}/mdl.pth')
        # torch.save() # for scheduler optimizer state

    model.save(f'{log_dir}/mdl.pth')

    # from val import validation
    # ds = LPDataset(args.dataset_processed_prefix,
    #                chunk=None, transform=MyToBipartite(phase='val')
    #                )
    # train_ds, val_ds = split_train_val(ds,args.seed)
    # kwargs = dict(batch_size=1, num_workers=num_workers,
    #               shuffle=False, )
    # val_loader = DataLoader(val_ds, **kwargs)
    # train_loader = DataLoader(train_ds, **kwargs)

    # avg_loss, avg_acc = validation(model, val_loader, dev)
    # print(avg_loss, 'avg val acc', avg_acc)
    # avg_loss, avg_acc = validation(model, train_loader, dev)
    # print(avg_loss, 'avg train acc', avg_acc)

if __name__ == '__main__':
    args = utils.Environment(
    )
    run_exp(args) 
