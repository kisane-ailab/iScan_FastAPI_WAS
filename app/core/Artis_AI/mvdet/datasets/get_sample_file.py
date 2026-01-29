import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm 
import subprocess


if __name__=='__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dirname = '/SSDe/kisane_DB/kisane_DB_v0_3/'
    savedir = './sample'
    os.makedirs(savedir, exist_ok=True)
    
    ann_path = os.path.join(dirname, 'val_multi.json')
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    
    for img in tqdm(ann['images']):
        fname = img['file_name']
        old = os.path.join(dirname, fname)
        new = os.path.join(savedir, os.path.basename(fname))
        subprocess.call('cp %s %s' %(old, new), shell=True)
    