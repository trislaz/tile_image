"""
Implements a class that is able to output the best representative tile, 
given an average one.
"""
from glob import glob
import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from openslide import open_slide

class TileRepresentant:
    """TileRepresentant.
    Implements methods to retrieve representative image tiles. 
    Implemented for the nearest neighbor tile."""

    def __init__(self, path_embed, path_raw, N=500):
        """__init__.

        :param path_embed: str: path to the embedded tiles.
        :param path_raw: path to the raw images (.ndpi, .tiff)
        :param N: number of tiles per image to sample to create the nearest 
        neighbors pool.
        """
        self.path_embed = path_embed
        self.path_raw = path_raw
        self.N = N
        self.emb, self.dicts = self.create_pool_of_tiles(path_embed)
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.emb)

    def create_pool_of_tiles(self, path_embed):
        pool_embeddings = []
        pool_dicts = []
        embeddings = glob(os.path.join(self.path_embed, 'mat_pca', '*.npy')) 
        for e in embeddings:
            name, _ = os.path.splitext(os.path.basename(e))
            mat = np.load(e)
            n_sample = min(self.N, mat.shape[0])
            indices = np.random.randint(mat.shape[0], n_sample)
            info_path = os.path.join(self.path_embed, 'info', name)
            with open(info_path, 'rb') as f:
                infodict = self.add_name_to_dict(pickle.load(f), name)
            pool_embeddings.append(mat[indices, :])
            pool_dicts += list(np.array(infodict)[indices])
        return np.vstack(pool_embeddings), pool_dicts
    
    def add_name_to_dict(self, dico, name):
        """
        because the name of the slide is not included in the dico.
        """
        for d in dico:
            d['name'] = name
        return d

    def get_nn_params(self, average_tile:list):
        """get_nn_params. get nearest neighbors params
        
        :param average_tiles:
        :type average_tiles: list
        """
        _, indices = self.nn.kneighbors(average_tile)
        param = np.array(self.dicts)[indices]
        return param

    def get_image(self, average_tile):
        param = self.get_nn_params(average_tile)
        path_wsi = os.path.join(self.path_raw, param['name'] + '.*')
        assert path_wsi, "no wsi with name {}".format(param['name'])
        assert len(path_wsi)<2, "several wsi with name {}".format(param['name'])
        slide = open_slide(path_wsi[0])
        image = slide.read_region(location=(param['x'], param['y']),
                                level=param['level'], 
                                size=(param['xsize'], param['ysize']))
        return image







