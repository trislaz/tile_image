import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.morphology import square, closing, opening
from skimage.segmentation import clear_border
from skimage import measure
import useful_wsi as usi
import os


#modif : selection of one of the biggest connected components.

def main(image, out, mask_level):
    slide = usi.open_image(image)
    if mask_level < 0:
        mask_level = len(slide.level_dimensions) + mask_level
    im = usi.get_whole_image(slide, level=mask_level, numpy=True) 
    num_histo, _ = os.path.splitext(os.path.basename(image))
    im_gray = rgb2gray(im)
    im_gray = clear_border(im_gray, prop=30)
    size = im_gray.shape
    im_gray = im_gray.flatten()
    pixels_int = im_gray[np.logical_and(im_gray > 0.1, im_gray < 0.98)]
    t = threshold_otsu(pixels_int)
    mask = opening(closing(np.logical_and(im_gray<t, im_gray>0.1).reshape(size), selem=square(2)), selem=square(2))
    print( 'mask  ',mask.sum())
    mask_path = os.path.join(out, num_histo+'.npy')
    final_mask = mask
    #if mask.sum() >= (mask.shape[0]*mask.shape[1])/25:
    #    labeled_c = measure.label(mask, background=0, connectivity=2)
    #    size = [(labeled_c==(x+1)).sum() for x in range(labeled_c.max())]
    #    biggest_cc = (labeled_c == np.argmax(size)+1).astype(int)
    #    print('biggest cc   ',biggest_cc.sum())
    #    final_mask = biggest_cc
    np.save(mask_path, final_mask)

def clear_border(mask, prop):
    r, c = mask.shape
    pr, pc = r//prop, c//prop
    mask[:pr, :] = 0
    mask[r-pr:, :] = 0
    mask[:,:pc] = 0
    mask[:,c-pc:] = 0
    return mask
