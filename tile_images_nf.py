#%%
import useful_wsi as usi
from glob import glob
from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
from xml.dom import minidom
from tqdm import tqdm
from skimage.draw import polygon
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage._shared.utils import warn


def threshold_otsu(image, mask, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    raveled_image = image[mask > 0].ravel()
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    if raveled_image.min() == raveled_image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(raveled_image.min()))

    hist, bin_centers = histogram(raveled_image, nbins)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold




def get_polygon(image, path_xml, label):
    
    doc = minidom.parse(path_xml).childNodes[0]
    nrows = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('nrows')[0].firstChild.data
    ncols = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('ncols')[0].firstChild.data
    size_image = (image.shape[0], image.shape[1])
    mask = np.zeros(size_image)
    obj = doc.getElementsByTagName('object')
    polygons = []
    for o in obj:
        if True:#o.getElementsByTagName('name')[0].firstChild.data == label:
            polygons += o.getElementsByTagName('polygon')
            print(polygons)
    if not polygons:
        raise ValueError('There is no annotation with label {}'.format(label))

    for poly in polygons:
        rows = []
        cols = []
        for point in poly.getElementsByTagName('pt'):
            x = int(point.getElementsByTagName('x')[0].firstChild.data)
            y = int(point.getElementsByTagName('y')[0].firstChild.data)
            rows.append(y)
            cols.append(x)
        rr, cc = polygon(rows, cols)
        mask[rr, cc] = 1
    return mask

def make_label_with_otsu(xml_file, rgb_img):
    mask = get_polygon(rgb_img, xml_file,label='t')
    grey_rgb = rgb2gray(rgb_img)
    thresh = threshold_otsu(grey_rgb, mask, nbins=256)
    binary = (grey_rgb < thresh).astype(float)
    merge_mask = mask + binary
    merge_mask[merge_mask != 2] = 0
    merge_mask[merge_mask > 0] = 1
    merge_mask = dilation(merge_mask)
    return merge_mask



#%% Downsizing an image
def tile_image(wsi, xml, level, size, auto_mask):
    """Downsize at level and save a WSI in the output folder 
    
    Parameters
    ----------
    wsi : str
        path to the WSI. May also be an openslide image.
    outfolder : str
        path where to stock the image
    level : int
        level to which downsample
    """
    name_wsi, ext = os.path.splitext(os.path.basename(wsi))
    slide = usi.open_image(wsi)
    mask_level = slide.level_count - 2
    rgb_img = usi.get_whole_image(slide, level=mask_level, numpy=True)
    if xml == 'no':
        mask_function = lambda x: np.ones(rgb_img.shape[0:-1])
    else:
        if auto_mask:
            name_xml = os.path.join(xml, name_wsi + ".npy")   
            mask_function = lambda x: np.load(name_xml)
        else:
            name_xml = os.path.join(xml, name_wsi + ".xml")   
            mask_function = lambda x: get_polygon(image=rgb_img, path_xml=name_xml, label='t')
    param_tiles = usi.patch_sampling(slide=wsi, mask_level=mask_level, mask_function=mask_function, sampling_method='grid', analyse_level=level, patch_size=(size,size), mask_tolerance = 0.8)
    for o, para in enumerate(param_tiles):
        patch = usi.get_image(slide=wsi, para=para, numpy=False)
        patch = patch.convert('RGB')
        new_name =  "tile_{}.jpg".format(o)
        patch.save(new_name)


#%% parser
parser = ArgumentParser()
parser.add_argument('--path', required=True, type=str, help="path of the files to downsample (tiff or svs files)")
parser.add_argument('--xml', type=str, default='no', help='either a path to the xml file, if no, then the whole image is tiled')
parser.add_argument('--level', type=int, default = 2, help="scale to which downsample. I.e a scale of 2 means dimensions divided by 2^2")
parser.add_argument('--size', type=int, default = 256, help="size of patches")
parser.add_argument('--auto_mask', type=int, default=1, help="if 1, mask is .npy, .xml else")
args = parser.parse_args()

#%% get the files
tiff = glob(os.path.join(args.path, "*.tiff"))
svs = glob(os.path.join(args.path, "*.svs"))
files = tiff + svs

if args.path.endswith(".tiff"):
    name_f, _ = os.path.splitext(os.path.basename(args.path))
    tile_image(args.path, args.xml, args.level, args.size, args.auto_mask)


