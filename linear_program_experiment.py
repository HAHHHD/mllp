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

####################################
#             config               #
####################################

cfg = load_config()
device = torch.device('cpu')
set_seed()
#device = torch.device('cpu')

####################################
#            training              #
####################################

if cfg.train_data_type == 'random':
    train_dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 1)
elif cfg.train_data_type == 'netlib':
    if cfg.methods[0] == 'invariant':
        train_dataset, train_dict = get_netlib_dataset_dense(normalize=True)
    else:
        train_dataset, train_dict = get_netlib_dataset(normalize=True)
elif cfg.train_data_type == 'twitch':
    train_dataset = get_twitch_dataset()
else:
    raise ValueError(f'Unknown training dataset {cfg.train_data_type}!')

criterion = torch.nn.BCEWithLogitsLoss()
#metric = BinaryF1Score()
#model = GNNModel().to(device)

for method_name in cfg.methods:
    model_path = f'linear_program_{cfg.train_data_type}_{method_name}.pt'
    # not os.path.exists(model_path) and
    if method_name in ['invariant']:
        print(f'Training the model weights for {method_name}...')
        model = InvariantModel(feat_dim = 50, depth = 2).to(device)
        for name, params in list(model.named_parameters()):
            print(name, params.shape)
        train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
        for epoch in range(cfg.train_iter):
            obj_sum = 0

            #print(model.linear[0].data)

            for name, constr_Q, coefs, basis_opt in train_dataset:
                X = torch.FloatTensor(constr_Q).to(device)
                coefs = torch.FloatTensor(coefs).to(device)
                latent_vars = model(X, coefs)
                basis_opt = torch.tensor(basis_opt, dtype = torch.float, device = device)
                obj = criterion(latent_vars, basis_opt)
                obj.backward()
                obj_sum += obj.mean()
                #print(model.linear[0].data)
                #print(model.linear[0].grad)
                #print(latent_vars)
                train_optimizer.step()
                train_optimizer.zero_grad()

                pred_indices_k_1 = torch.topk(latent_vars, k=1+constr_Q.shape[1])[-1].cpu().detach().numpy()
                pred_indices = pred_indices_k_1[:constr_Q.shape[1]]
                print(latent_vars[pred_indices_k_1[-1]])
                print(torch.min(latent_vars[pred_indices]))
                print(latent_vars[pred_indices_k_1[-2]])
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
