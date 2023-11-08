import os
import scipy
import numpy as np
import gurobipy as gp
import torch
import time
path = "processed/ftp"
files = os.listdir(path)
os.makedirs(path+"p", exist_ok = True)
os.makedirs(path+"s", exist_ok = True)
check_list = []

for file in files:
    print("======")
    if file == '1.mps':
        continue
    print(file)
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
    #model.Params.Presolve = 2
    model.Params.Method = 0
    model = model.presolve()
    model.write(path+"p/"+file)
