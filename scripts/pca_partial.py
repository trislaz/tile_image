#%% code written by tristan lazard
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
from joblib import load, dump
import os

def check_dim(batch):
    """ Checks if batch is big enough for the incremental PCA to be 
    efficient.
    
    Parameters
    ----------
    batch : list
        list of matrix, each matrix corresponding to a WSI divided in $row tiles
    
    Returns
    -------
    bool
        Is the batch big enough ?
    """
    if batch:
        n_tiles = np.sum([x.shape[0] for x in batch])
        n_features = batch[-1].shape[1]
        ans = n_tiles >= n_features
    else:
        ans = False
    return ans

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default=".", help="path to the files of tiles")
    args = parser.parse_args()
    files = glob(os.path.join(args.path, "*.npy"))
    ipca = IncrementalPCA()
    batch = []
    for path in tqdm(files):
        mat = np.load(path)
        if mat.sum() == 0:
            continue
        if check_dim(batch):
            batch = np.concatenate(batch, axis=0)
            ipca.partial_fit(X=batch)
            batch = []
        else:
            batch.append(mat)

    msg = " ----------------  RESULTS -------------------- \n"
    s = 0
    for i,o in enumerate(ipca.explained_variance_ratio_, 1):
        s += o
        msg += "Dimensions until {} explains {}% of the variance \n".format(i, s*100)
    msg += "----------------------------------------------"

    ## Saving
    with open('results.txt', 'w') as f:
        f.write(msg)

    dump(ipca, 'pca_tiles.joblib')
