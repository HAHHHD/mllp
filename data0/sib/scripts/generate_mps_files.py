import os, sys
sys.path.insert(0, os.path.dirname(__file__) + '/../')
from scripts.run_cplex import lp_to_mps
from utils import shell 

DS= 'facilities' # 'indset' #
pref='/cache/lp-dataset/'
home=os.environ['HOME']
cplex_cmd = f'{home}/work/CPLEX/cplex/bin/x86-64_linux/cplex'
lp_path = f'{pref}/{DS}/lp/'
mps_path = f'{pref}/{DS}/mps/'
shell(f'mkdir -p {mps_path}') 
os.system('gzip -d {}lp_Sample_*.lp.gz'.format(lp_path))

for file in os.listdir(lp_path):
    if file.endswith(".lp"):
        # print('proc ', file)
        # Replace '+ -' by '-' in lp file
        # os.system("sed -i -e 's/+ -/-/g' {}{}".format(lp_path, file))
        lp_to_mps(lp_path, file, mps_path, cplex_cmd)

print('done!')
