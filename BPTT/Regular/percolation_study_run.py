import torch
import numpy as np
import dill
import os
import sys

dir_name = sys.argv[2]
gpu_id = int(sys.argv[1])
torch.cuda.set_device(gpu_id)
print('GPU: ', torch.cuda.current_device())

srnns_files = [f for f in os.listdir(dir_name) 
                if os.path.isfile(os.path.join(dir_name, f)) and 
                   int(f[-5])==gpu_id and 
                   f[0]!='t' and 
                   ('trained_%s'%f not in os.listdir(dir_name) or len(sys.argv)==4)]
srnns_files.sort()

rerun_id = '' if len(sys.argv)<4 else '_'+sys.argv[3]

for srnn_file in srnns_files:
    print('######################## %s ########################'%srnn_file)
    srnn = dill.load(open('%s/%s'%(dir_name, srnn_file), 'rb'))
    srnn.train()
    dill.dump(srnn, open('%s/trained%s_%s'%(dir_name, rerun_id, srnn_file), 'wb'))
    del srnn
