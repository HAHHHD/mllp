import faulthandler
# 在import之后直接添加以下启用代码即可
print("ssss")
faulthandler.enable()
from linear_program_methods import *
import time
#import xlwt
from datetime import datetime
import torch_geometric as pyg
import os
import sys
import json
from config import load_config
from linear_program_data import get_random_dataset, get_twitch_dataset, get_facebook_dataset, get_netlib_dataset, get_netlib_dataset_dense
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

####################################
#             config               #
####################################
#-----------------------*********
torch.cuda.set_device(0)
cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type) 
set_seed()
#device = torch.device('cpu')

####################################
#            training              #
####################################

def get_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return allocated, reserved


if cfg.train_data_type == 'random':
    train_dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 1)
elif cfg.train_data_type == 'netlib':
    if cfg.methods[0] == 'invariant':
        train_dataset, train_dict = get_netlib_dataset_dense(normalize=True)
    if cfg.methods[0] == 'angleNet':
        #train_dataset, train_dict = get_netlib_dataset_dense(normalize=True)
        #train_loader = get_netlib_dataloader(train_dataset, device)
        train_loader, train_dict= get_netlib_dataloader_pack(True, device)
    else:
        train_dataset, train_dict = get_netlib_dataset(normalize=True)
elif cfg.train_data_type == 'twitch':
    train_dataset = get_twitch_dataset()
else:
    raise ValueError(f'Unknown training dataset {cfg.train_data_type}!')


@torch.no_grad() 
def labels_to_balanced_weights(labels, merge_lu=True): 
    ## merge_lu for two-sides 
    #print(labels)
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

def balanced(logpr_cons, logpr_vars, y_s, y_t): 
    loss=0. 
    m,n = len(y_s), len(y_t)
    criterion=torch.nn.CrossEntropyLoss(weight=labels_to_balanced_weights(y_s))  
    loss+=(m+n)/m*criterion(logpr_cons, y_s) 
    criterion=torch.nn.CrossEntropyLoss(weight=labels_to_balanced_weights(y_t))   
    loss+=(m+n)/n*criterion(logpr_vars, y_t)  
    return loss 

criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss()
#criterion = torch.nn.BCELoss()
#metric = BinaryF1Score()
#model = GNNModel().to(device)

for method_name in cfg.methods:
    model_path = f'linear_program_{cfg.train_data_type}_{method_name}.pt'
    # not os.path.exists(model_path) and
    if method_name in ['invariant']:
        print(f'Training the model weights for {method_name}...')
        model = InvariantModel(feat_dim = 50, depth = 2).to(device)
        for params in list(model.parameters()):
            print(params.shape)

        train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
        for epoch in range(cfg.train_iter):
            obj_sum = 0
            for name, constr_Q, coefs, basis_opt in train_dataset:
                X = torch.FloatTensor(constr_Q).to(device)
                coefs = torch.FloatTensor(coefs).to(device)
                latent_vars = model(X, coefs)
                print(latent_vars)
                basis_opt = torch.tensor(basis_opt, dtype = torch.float, device = device)
                obj = criterion(latent_vars, basis_opt)
                obj.backward()
                obj_sum += obj.mean()
                train_optimizer.step()
                train_optimizer.zero_grad()

                pred_indices = torch.topk(latent_vars, k=constr_Q.shape[1])[-1].cpu().detach().numpy()
                pred = np.zeros([coefs.shape[0]])
                pred[pred_indices] = 1
                f1 = f1_score(basis_opt.cpu().detach().numpy(), pred)
                correct_num = pred @ basis_opt.cpu().detach().numpy()
                #print(correct_num, rhs.shape[0], rhs.shape[0] + coefs.shape[0])
                print(f'%8d, %8d, %8d, %5f'%(correct_num, constr_Q.shape[1], coefs.shape[0], f1))
                train_dict[name].append(correct_num)
            train_dict["obj"].append(obj_sum.cpu().detach().numpy() / len(train_dataset))
            with open("train_log.json", "w") as json_file:
                json.dump(train_dict, json_file)
            print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
    elif method_name == "angleNet":
        print(f'Training the model weights for {method_name}...')
        model = AngleModel(feat_dim = 256).to(device)
        #for params in list(model.parameters()):
        #    print(params.shape)

        train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr, weight_decay=1e-4)
        scheduler = StepLR(train_optimizer, step_size=max(cfg.train_iter // 4, 1), gamma=0.1)
        #scheduler = MultiStepLR(train_optimizer, milestones=[10, 100, 200, 300, 400, 500, 600, 700], gamma=0.3)

        best_acc = 0
        dic = {"epoch":[], "best_acc":[]}
        for epoch in range(cfg.train_iter):
            model.train()
            obj_sum = 0
            #for name, constr_Q, coefs, basis_opt in train_dataset:
            avg_f1 = 0
            avg_acc = 0
            skip = 0
            thred = 10000000
            for _, batched_graphs in enumerate(train_loader, 0):
                acc = 0
                leng = 0
                batched_graphs.y = batched_graphs.y[0]
                batched_graphs.is_var = batched_graphs.is_var[0]
                if (sum(np.array(batched_graphs.y == 3)) > 0): # skip special instances 
                        skip += 1
                        continue
                basis_num, var_num = batched_graphs.basis_num[0], batched_graphs.var_num[0]
                #print(batched_graphs.y.dtype, batched_graphs.is_var.dtype, name.dtype, basis_num.dtype, var_num.dtype)
                if (var_num + 1 < thred):
                    loader = [batched_graphs]
                else: # neighborloader not fitted yet
                    batch_size = thred
                    loader = NeighborLoader(batched_graphs, input_nodes=None, directed=False,
                                            num_neighbors=[6]*2, shuffle=True, batch_size=batch_size,
                                            num_workers=1, pin_memory=True,
                                            drop_last=False
                                            )
                #for graph in tqdm(loader):
                for graph in loader:
                    graph.to(device)
                    #name, basis_num, var_num, basis_opt = graph.name[0], graph.basis_num[0], graph.var_num[0], torch.tensor(graph.basis_opt[0], dtype = torch.float, device = device)
                    basis_opt= torch.tensor(graph.y, dtype=torch.long, device=device)
                    train_optimizer.zero_grad()
                    #print("var num", var_num)
                    torch.cuda.empty_cache()
                    #allocated, reserved = get_gpu_memory_usage()
                    #print("Allocated:", allocated, "bytes")
                    #print("Reserved:", reserved, "bytes")
                    '''if hasattr(graph, 'batch_size'): # neighborloader
                        latent_vars = model(graph)[:graph.batch_size]
                        basis_opt = basis_opt[:graph.batch_size]
                        print(graph)
                        #print(graph.num_sampled_nodes)
                    else:
                        latent_vars = model(graph)'''
                    latent_vars = model(graph)
                    #---------------------------- for general form ------------------------------------------
                    #print(basis_opt.dtype)
                    #basis_opt = basis_opt.to(torch.long)
                    #print(latent_vars, basis_opt)
                    obj = balanced(latent_vars[var_num - basis_num:], latent_vars[:var_num - basis_num], basis_opt[var_num - basis_num:], basis_opt[:var_num - basis_num])
                    #obj = criterion(latent_vars, basis_opt)
                    if (torch.isnan(obj).item()):
                        print(graph, name, basis_num, var_num. basis_opt)
                    assert not torch.isnan(obj).item()
                    obj.backward()
                    obj_sum += obj.item()
                    train_optimizer.step()
                    del obj

                    with torch.no_grad():
                        basis_num0 = np.sum(basis_opt.detach().cpu().numpy() == 1)
                        pr = F.softmax(latent_vars.float(), dim=-1) 
                        #pr = pr.clone()
                        pr[torch.isnan(pr)]=0 # if half model, nan will occur 

                        _, topk_idx = pr[:, 1].topk(basis_num0)
                        pr[:, 1] = pr.min() - 1
                        pr[topk_idx, 1] = pr.max() + 1
                        pred = pr.argmax(-1)
                        del _
                        del topk_idx
                        del pr
                        pred = pred.to(torch.long).cpu().detach().numpy()
                        basis_opt = basis_opt.cpu().detach().numpy()
                        torch.cuda.empty_cache()

                    f1 = 0
                    assert (np.sum(pred == 1) == np.sum(basis_opt == 1))
                    acc += np.sum(pred == basis_opt)
                    leng += basis_opt.shape[0]
                    del pred
                    del basis_opt
                    del latent_vars
                    torch.cuda.empty_cache()
                    #---------------------------- for general form ------------------------------------------

                    #---------------------------- for standard form ------------------------------------------
                    '''obj = criterion(latent_vars, basis_opt)
                    assert not torch.isnan(obj).item()
                    obj.backward()
                    #print(obj.mean())
                    obj_sum += obj.mean()
                    train_optimizer.step()

                    pred_indices = torch.topk(latent_vars, k=basis_num)[-1].cpu().detach().numpy()
                    pred = np.zeros(var_num)
                    pred[pred_indices] = 1
                    f1 = f1_score(basis_opt.cpu().detach().numpy(), pred)
                    acc += pred @ basis_opt.cpu().detach().numpy()'''
                    #---------------------------- for standard form ------------------------------------------

                acc /= var_num.item()
                #print(leng, var_num)
                avg_f1 += f1
                avg_acc += acc
                #print(latent_vars)
                #print(obj)
                #print(correct_num, rhs.shape[0], rhs.shape[0] + coefs.shape[0])
                #print(f'%8d, %8d, %8d, %5f'%(correct_num, basis_num, var_num, f1))
                print(f'%8d, %8d, %5f'%(basis_num, var_num, acc))
                #train_dict[name].append(correct_num)
            train_dict["obj"].append(obj_sum / (len(train_loader) - skip))
            #train_dict["acc"].append(avg_acc.cpu().detach().numpy() / len(train_loader))
            train_dict["acc"].append(avg_acc / (len(train_loader) - skip))
            #*********----------------*********
            with open("train_log_#00444###.json", "w") as json_file:
                json.dump(train_dict, json_file)
            print(f'epoch {epoch}, len={len(train_loader)}, skip={skip}, obj={obj_sum / (len(train_loader) - skip)}, avg_f1={avg_f1 / (len(train_loader) - skip)}, avg_acc={avg_acc / (len(train_loader) - skip)}')
            if avg_acc / (len(train_loader) - skip) > best_acc:
                best_acc = avg_acc / (len(train_loader) - skip)
                #*********-----------------*********
                torch.save(model, 'model_#00444###.pth')
                print("------------------------best_acc------------------------")
                #print(f'epoch {epoch}, len={len(train_loader)}, obj={obj_sum / len(train_loader)}, avg_f1={avg_f1 / len(train_loader)}, avg_acc={avg_acc / len(train_loader)}')
                print(f'epoch {epoch}, len={len(train_loader)}, skip={skip}, obj={obj_sum / (len(train_loader) - skip)}, avg_acc={avg_acc / (len(train_loader) - skip)}')
                #*********---------------*********
                with open("best_acc_#00444###.json", "w") as json_file:
                    #dic["best_acc"].append((epoch, obj_sum.cpu().detach().numpy() / len(train_loader), best_acc.cpu().detach().numpy().tolist()))
                    dic["best_acc"].append((epoch, obj_sum / (len(train_loader) - skip), best_acc.tolist()))
                    json.dump(dic, json_file)
            scheduler.step()
            del obj_sum 
            del avg_acc
            torch.cuda.empty_cache()



    elif method_name in ['gs-topk', 'soft-topk', 'egn']:
        print(f'Training the model weights for {method_name}...')
        model = GNNModel().to(device)
        if method_name in ['gs-topk', 'soft-topk']:
            train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
            for epoch in range(cfg.train_iter):
                # training loop
                obj_sum = 0
                for name, constrs, constr_weights, coefs, rhs, basis_opt in train_dataset:
                    graph = build_graph_from_weights_sets(constrs, constr_weights, rhs, coefs, device)
                    latent_vars = model(graph)
                    if method_name == 'gs-topk':
                        sample_num = cfg.train_gumbel_sample_num
                        noise_fact = cfg.gumbel_sigma
                    else:
                        sample_num = 1
                        noise_fact = 0
                    """
                    top_k_indices, probs = gumbel_sinkhorn_topk(
                        latent_vars, len(constrs), max_iter=cfg.sinkhorn_iter, tau=cfg.sinkhorn_tau,
                        sample_num=sample_num, noise_fact=noise_fact, return_prob=True
                    )
                    """
                    #obj, _ = compute_obj_differentiable(weights, sets, probs, bipartite_adj, device=probs.device)
                    basis_opt = torch.tensor(basis_opt, dtype = torch.float, device = device)
                    obj = criterion(latent_vars, basis_opt)
                    obj.backward()
                    obj_sum += obj.mean()
                    train_optimizer.step()
                    train_optimizer.zero_grad()

                    pred_indices = torch.topk(latent_vars, k=rhs.shape[0])[-1].cpu().detach().numpy()
                    pred = np.zeros([coefs.shape[0]])
                    pred[pred_indices] = 1
                    f1 = f1_score(basis_opt.cpu().detach().numpy(), pred)
                    correct_num = pred @ basis_opt.cpu().detach().numpy()
                    #print(correct_num, rhs.shape[0], rhs.shape[0] + coefs.shape[0])
                    print(f'%8d, %8d, %8d, %5f'%(correct_num, rhs.shape[0], coefs.shape[0], f1))
                    train_dict[name].append(correct_num)
                train_dict["obj"].append(obj_sum.cpu().detach().numpy() / len(train_dataset))
                with open("train_log.json", "w") as json_file:
                    json.dump(train_dict, json_file)
                print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
        if method_name in ['egn']:
            train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
            # training loop
            for epoch in range(cfg.train_iter):
                obj_sum = 0
                for name, weights, sets in train_dataset:
                    bipartite_adj = None
                    graph = build_graph_from_weights_sets(weights, sets, device)
                    probs = model(graph)
                    constraint_conflict = torch.relu(probs.sum() - cfg.train_max_covering_items)
                    obj, _ = compute_obj_differentiable(weights, sets, probs, bipartite_adj, device=probs.device)
                    obj = obj - cfg.egn_beta * constraint_conflict
                    (-obj).mean().backward()
                    obj_sum += obj.mean()

                    train_optimizer.step()
                    train_optimizer.zero_grad()
                print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}.')

sys.exit()
####################################
#            testing               #
####################################

if cfg.test_data_type == 'random':
    dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 0)
elif cfg.test_data_type == 'facebook':
    dataset = get_facebook_dataset()
elif cfg.test_data_type == 'twitch':
    dataset = get_twitch_dataset()
else:
    raise ValueError(f'Unknown testing dataset {cfg.test_data_type}!')

wb = xlwt.Workbook()
ws = wb.add_sheet(f'max_covering_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}')
ws.write(0, 0, 'name')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

torch.random.manual_seed(1)
for index, (name, weights, sets) in enumerate(dataset):
    method_idx = 0
    print('-' * 20)
    print(f'{name} items={len(weights)} sets={len(sets)}')
    ws.write(index+1, 0, name)

    # greedy
    if 'greedy' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        objective, selected = greedy_max_covering(weights, sets, cfg.test_max_covering_items)
        print(f'{name} greedy objective={objective} selected={sorted(selected)} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'greedy-objective')
            ws.write(0, method_idx*2, 'greedy-time')
        ws.write(index+1, method_idx*2-1, objective)
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    # SCIP - integer programming
    if 'scip' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = ortools_max_covering(weights, sets, cfg.test_max_covering_items, solver_name='SCIP', linear_relaxation=False, timeout_sec=cfg.solver_timeout)
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        top_k_indices = sorted(top_k_indices.cpu().numpy().tolist())
        print(f'{name} SCIP objective={ip_obj:.0f} selected={top_k_indices} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'SCIP-objective')
            ws.write(0, method_idx*2, 'SCIP-time')
        ws.write(index+1, method_idx*2-1, ip_obj)
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    # Gurobi - integer programming
    if 'gurobi' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = gurobi_max_covering(weights, sets, cfg.test_max_covering_items, linear_relaxation=False, timeout_sec=cfg.solver_timeout, verbose=cfg.verbose)
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        top_k_indices = sorted(top_k_indices.cpu().numpy().tolist())
        print(f'{name} Gurobi objective={ip_obj:.0f} selected={top_k_indices} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'Gurobi-objective')
            ws.write(0, method_idx*2, 'Gurobi-time')
        ws.write(index+1, method_idx*2-1, ip_obj)
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    # Erdos Goes Neural
    if 'egn' in cfg.methods:
        method_idx += 1
        model.load_state_dict(torch.load(f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_egn.pt'))
        objective, best_top_k_indices, finish_time = egn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.egn_beta)
        print(f'{index} egn objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'egn-objective')
            ws.write(0, method_idx * 2, 'egn-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

        method_idx += 1
        objective, best_top_k_indices, finish_time = egn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.egn_beta, cfg.egn_trials)
        print(f'{index} egn-accu objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'egn-accu-objective')
            ws.write(0, method_idx * 2, 'egn-accu-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    # SOFT-TopK
    if 'soft-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_soft-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(weights, sets, cfg.test_max_covering_items, model, 1, 0, cfg.sinkhorn_tau, cfg.sinkhorn_iter, cfg.soft_opt_iter, verbose=cfg.verbose)
        print(f'{name} SOFT-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'SOFT-TopK-objective')
            ws.write(0, method_idx * 2, 'SOFT-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    # GS-TopK
    if 'gs-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_gs-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.gumbel_sample_num, cfg.gumbel_sigma, cfg.sinkhorn_tau, cfg.sinkhorn_iter, cfg.gs_opt_iter, verbose=cfg.verbose)
        print(f'{name} GS-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'GS-TopK-objective')
            ws.write(0, method_idx * 2, 'GS-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-prev_time)

        # Homotopy-GS-TopK
        method_idx += 1
        prev_time = time.time()
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.gumbel_sample_num, cfg.homotophy_sigma, cfg.homotophy_tau, cfg.homotophy_sk_iter, cfg.homotophy_opt_iter, verbose=cfg.verbose)
        print(f'{name} Homotopy-GS-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'Homotopy-GS-TopK-objective')
            ws.write(0, method_idx * 2, 'Homotopy-GS-TopK-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # perturb-TopK
    if 'perturb-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_gs-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = gumbel_max_covering(weights, sets, cfg.test_max_covering_items, model,
                                                           cfg.gumbel_sample_num * 10, # needs 10x more samples than others
                                                           cfg.perturb_sigma, cfg.perturb_opt_iter, verbose=cfg.verbose)
        print(f'{name} perturb-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'perturb-TopK-objective')
            ws.write(0, method_idx * 2, 'perturb-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    # blackbox-TopK
    if 'blackbox-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_gs-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = blackbox_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.blackbox_lambda, cfg.blackbox_opt_iter, verbose=cfg.verbose)
        print(f'{name} blackbox-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'blackbox-TopK-objective')
            ws.write(0, method_idx * 2, 'blackbox-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-prev_time)

    # LML-TopK
    if 'lml-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_gs-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = lml_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.lml_opt_iter, verbose=cfg.verbose)
        print(
            f'{name} LML-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'LML-TopK-objective')
            ws.write(0, method_idx * 2, 'LML-TopK-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    wb.save(f'max_covering_result_{cfg.test_data_type}_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_{timestamp}.xls')
