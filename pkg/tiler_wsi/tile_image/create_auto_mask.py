from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.morphology import square, closing, opening
from skimage.segmentation import clear_border
from argparse import ArgumentParser
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage import img_as_float, img_as_ubyte



class MaskOverlay:
    color_dict = {'r':0, 'g':1, 'b':2}
    def __init__(self, binary_mask=True, color = 'r', alpha = 0.6):
        self.color = color
        self.alpha = alpha
        self.binary_mask = binary_mask
        self.im = None  # Implemented in call ..
        self.mask = None
        self.im_masked = None

    def __call__(self, im, mask):
        im, mask = img_as_float(im), img_as_float(mask)
        self.mask = mask
        if self.binary_mask:
            mask = self.binary_to_rgb(mask)
        if len(im.shape) < 3:
            im = self.gray_to_rgb(im)
        im_hsv = rgb2hsv(im)
        mask_hsv = rgb2hsv(mask)
        im_hsv[..., 0] = mask_hsv[..., 0] # Gives the mask color
        im_hsv[..., 1] = mask_hsv[..., 1]*self.alpha # Gives the "transparancy"
        im_masked = hsv2rgb(im_hsv)
        self.im = im
        self.im_masked = im_masked
        return im_hsv

    def binary_to_rgb(self, mask):
        l, c = mask.shape
        rgb_mask = np.zeros((l, c, 3))
        color = self.color_dict[self.color]
        rgb_mask[:,:,color] = mask
        return rgb_mask

    def gray_to_rgb(self, im):
        im = np.dstack((im, im, im))
        return im

    def save_overlay(self, path):
        """saves the images overlayed by the mask as a jpg image
        
        Parameters
        ----------
        path : str
        """
        if self.im is None:
            raise ValueError('You didnt call the MaskOverlay, therefore there is no image in memory')
        im_masked = img_as_ubyte(self.im_masked)
        imsave(path, im_masked)

    def save_mask(self, path):
        """saves the mask as a numpy array of 1s and 0s.
        Saves the black/white version, not RGB !! dims = (H, L)
        
        Parameters
        ----------
        path : str
        """
        if self.im is None:
            raise ValueError('You didnt call the MaskOverlay, therefore there is no image in memory')
        np.save(path, self.mask)

    
def main():
    mo = MaskOverlay(color='b', alpha=0.3)
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, help="path to the image to treat")
    args = parser.parse_args()

    num_histo, _ = os.path.splitext(os.path.basename(args.image))
    im = imread(args.image)
    im_gray = rgb2gray(im)
    t = threshold_otsu(im_gray)
    mask = opening(closing(im_gray<t, selem=square(15)), selem=square(8))
    overlay_path = os.path.join(num_histo+'_visu.jpg')
    mask_path = os.path.join(num_histo+'.npy')
    _ =mo(im=im, mask=mask)
    mo.save_mask(mask_path)
    mo.save_overlay(overlay_path)

if __name__ == '__main__':
    main()
