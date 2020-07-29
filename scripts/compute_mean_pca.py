
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument("--path", 
                    type=str,
                    help="Path to the WSI file (.npy)")
args = parser.parse_args()

means = []
sizes = []
for file in tqdm(glob(os.path.join(args.path, '*.npy'))):
    tmp = np.load(file)
    sizes.append(tmp.shape[0])
    tmp = tmp.sum(axis=0)
    means.append(tmp)

means = np.array(means)
sizes = np.array(sizes)

mean = means.sum(axis=0) / sizes.sum()
np.save('mean.npy', mean)