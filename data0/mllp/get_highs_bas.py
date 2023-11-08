import os
import scipy
import numpy as np
import torch
import time


path = "processed/mps0000s"
pred_path = "pred/mps0000s/"
highs_path = "hp/mps0000s/"
files = os.listdir(path)
check_list = []
os.makedirs(highs_path, exist_ok = True)

for file in files:
    print("==============================================================================================================")
    vbas = np.load(pred_path+file+"_v.npy") + 1
    cbas = np.load(pred_path+file+"_c.npy") + 1
    f = open(highs_path+file+'.bas','w')
    f.write("HiGHS v1\nValid\n# Columns "+str(len(vbas))+"\n"+' '.join(map(str, vbas))+"\n# Rows "+str(len(cbas))+"\n"+' '.join(map(str, cbas)))
    f.close()