import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--path", 
                    type=str,
                    help="Path to the WSI file (.npy)")
args = parser.parse_args()

res = []
somme = None
nb_tiles = 0
for file in tqdm(glob('*.npy')):
    if somme is None:
        somme = np.zeros(file.shape[1])
    tmp = np.load(file)
    if tmp.sum() != 0:
        somme += tmp.sum(axis=0)
        nb_tiles += tmp.shape[0]

np.save('mean.npy', somme/nb_tiles)
