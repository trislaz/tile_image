
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--path", type=str, help="path where to find the *_embedded.npy matrices")

res = []
for file in tqdm(glob('*_embedded.npy')):
    tmp = np.load(file)
    res.append(tmp)

res_npy = np.vstack(res).mean(axis=0)
np.save('mean.npy', res_npy)