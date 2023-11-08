import os
import scipy
import numpy as np
import gurobipy as gp
import torch
import time
path = "processed/ftps"
dest_path = "dataset/ftps"
files = os.listdir(path)
os.makedirs(dest_path, exist_ok = True)
check_list = []

for file in files:
    print("======")
    if file == '1.mps':
        continue
    print(file)
    if (file == "prob_2.mps"):
        continue
    model = gp.read(path+"/"+file)
    #model.Params.Presolve = 0
    model.printStats()
    model.Params.AggFill = 0
    model.Params.Aggregate = 0
    model.Params.DualReductions = 0
    model.Params.PreCrush = 0
    model.Params.PreDual = 0
    model.Params.PreSparsify = 0
    model.Params.PreDepRow = 1
    model.Params.PreSOS1BigM = 0
    model.Params.PreSOS2BigM = 0
    model.Params.Presolve = 2
    model.Params.Method = 0
    #model = model.presolve()
    #model.write(path+"p/"+file)
    #if (len(model.getAttr("Obj",model.getVars()))) < 400000:
        #continue
    #p.Params.Method = 0
    #p.printStats()
    #p.optimize()
    model.Params.Presolve = 0
    model.optimize()
    basis_total = model.getA().shape[0]
    v_basis_list = np.array(model.getAttr("VBasis",model.getVars()))
    #print(v_basis_list)
    v_basis_list = (v_basis_list == 0).astype(np.int32)# + (v_basis_list == -3).astype(np.int32)
    #v_basis = len(v_basis_list) + np.sum(v_basis_list)
    #v_basis = np.sum(np.array(v_basis_list) == 0) + np,sum(np.array(v_basis_list) == -3)
    v_basis = np.sum(v_basis_list)
    c_basis_list = np.array(model.getAttr("CBasis",model.getConstrs()))
    #print(c_basis_list)
    #c_basis = len(c_basis_list) + np.sum(c_basis_list)
    c_basis_list = (c_basis_list == 0).astype(np.int32)
    #c_basis = np.sum(np.arrav(c_basis_list) == 0)
    c_basis = np.sum(c_basis_list)
    print(v_basis, c_basis)
    #check list.append(v_basis + c_basis == basis_total)
    assert(v_basis + c_basis == basis_total)
    assert(len(v_basis_list) == model.getA().shape[1])
    assert(len(c_basis_list) == model.getA().shape[0])
    constrs_mat = scipy.sparse.csc_matrix(model.getA().T)
    rhs = scipy.sparse.csc_matrix(np.expand_dims(model.getAttr("RHS",model.getConstrs()), axis=0))
    #print(constrs_mat, rhs)
    constrs_mat0 = scipy.sparse.vstack([constrs_mat, rhs], format="csc")
    ''''start = time.time()
    constrs_mat = constrs_mat0.todense()
    print(time.time() - start)
    start = time.time()
    constrs_mat = torch.tensor(constrs_mat, dtype=torch.float, device = 'cuda')
    print(time.time() - start)
    start = time.time()
    print(constrs_mat.shape)
    q, r = torch.linalg.qr(constrs_mat.T)
    print(time.time() - start)
    start = time.time()
    print(r)
    #print(constrs_mat.shape[1], torch.linalg.matrix_rank(r))
    print(time.time() - start)
    start = time.time()'''
    np.save(dest_path+"/"+file+"_v.npy", v_basis_list)
    np.save(dest_path+"/"+file+"_c.npy", c_basis_list)
    scipy.sparse.save_npz(dest_path+"/"+file+"_constrs.npz", model.getA())
    #print(constrs_mat0)
    scipy.io.mmwrite(dest_path+"/"+file+"_constrs", constrs_mat0)
    np.save(dest_path+"/"+file+"_rhs.npy", model.getAttr("RHS",model.getConstrs()))
    np.save(dest_path+"/"+file+"_coefs.npy", model.getAttr("Obj",model.getVars()))