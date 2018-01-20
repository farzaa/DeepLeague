import os
import numpy as np

def npz_stats(path):
    tot = 0
    for file_name in os.listdir(path):
        print(file_name)
        np_obj = np.load(path + file_name)
        tot = tot + len(np_obj['images'])
        print(tot)




npz_stats('/Volumes/DATA/clusters_cleaned/train/')
