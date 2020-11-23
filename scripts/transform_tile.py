#%% code written by tristan lazard
from argparse import ArgumentParser
from joblib import load
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def transform_pca(dataset, pca):
    """ Encodes the dataset in the basis of PCA.
        Centers the features.
    
    Parameters
    ----------
    dataset : np.array
        NxD, N samples with D features
    pca : sklearn.decomposition.PCA or sklearn.decomposition.IncrementalPCA 
        PCA fitted model.
    
    Returns
    -------
    np.array
        NxD', N samples, D' features = number of components of the PCA. = D.
    """
    if dataset.sum() != 0:
        data_transformed = np.matmul(dataset, pca.components_.transpose())
        data_transformed = data_transformed - pca.mean_
    else:
        data_transformed = dataset
    return data_transformed

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", 
                        type=str,
                        help="Path to the WSI file (.npy)")

    args = parser.parse_args()
    ipca = load(os.path.join(args.path, 'pca', 'pca_tiles.joblib'))
    files = glob(os.path.join(args.path, 'mat', '*.npy'))
    for f in files:
        name = os.path.basename(f)
        mat = np.load(f)
        mat_pca = ipca.transform(mat)
        np.save(name, mat_pca)

