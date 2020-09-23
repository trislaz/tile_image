#%%
#from .tile_functionnal import compute_distance
from scipy.ndimage import rotate, distance_transform_bf
import numpy as np
import pickle
import os 
        
class TileSampler:
    def __init__(self, wsi_path, info_folder):
        """Initialise a tile sampler object. Given a WSI, this sampler will automatize
        different ways of selecting tiles inside it. 
        This object is adapted to all types of WSI representations, as its sampling methods
        return indices of tiles instead of tiles themselves.

        Args:
            wsi_path (str): path to the whole slide image (ndpi, svs ...)
            info_folder (str): path to the info_folder of the dataset -and not of the slide-. 
        """
        name_wsi, _ = os.path.splitext(os.path.basename(wsi_path))
        name_wsi = name_wsi.split('_embedded')[0]
        self.name_wsi = name_wsi
        self.path_wsi = wsi_path
        path_infomat = os.path.join(info_folder, name_wsi + '_infomat.npy')
        self.infomat = np.load(path_infomat)
        self.mask = self.infomat > 0
        self.total_tiles = self.mask.sum()
        # is necessary to apply 0 to the border so that background is always surrounding the image.
        self.mask = self._force_background(self.mask)
        self.dist = distance_transform_bf(self.mask)
        path_infodict = os.path.join(info_folder, name_wsi + '_infodict.pickle')
        with open(path_infodict, 'rb') as f:
            self.infodict = pickle.load(f)

    @staticmethod
    def _force_background(mask):
        mask[0, :] = 0
        mask[-1,:] = 0
        mask[:,0] = 0
        mask[:,-1] = 0
        return mask

    def random_sampler(self, nb_tiles):
        indices = np.random.randint(self.total_tiles, size=nb_tiles)
        return indices

    def random_biopsie(self, nb_tiles):
        angle = np.random.randint(360)
        indices = self.artificial_biopsie(angle=angle, nb_tiles=nb_tiles)
        return indices
    
    def artificial_biopsie(self, angle, nb_tiles):
        """generate an artificial biopsie of $nb_tiles along the $angle.

        Args:
            angle (float): in degrees; angle of the biopsie
            nb_tiles (int): number of tiles to sample

        Returns:
            array: indices of the sampled tiles.
        """
        rotated_mask = rotate(self.mask, angle, order=0)
        rotated_dist = rotate(self.dist, angle, order=0)
        rotated_infomat = rotate(self.infomat, angle, order=0)
        _, y = np.unravel_index(rotated_dist.argmax(), rotated_dist.shape)
        sample = np.zeros(rotated_mask.shape)
        sample[:, y-1:y+3] = 1
        sample = sample.astype(bool)
        sample = sample & rotated_mask
        tile_indices = np.random.choice(rotated_infomat[sample], nb_tiles)
        tile_indices = tile_indices.astype(int)
        return tile_indices
