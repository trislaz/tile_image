#%%
import useful_wsi as usi
from glob import glob
from argparse import ArgumentParser
import os
from PIL import Image
from tqdm import tqdm

#%% Downsizing an image
def downsize(wsi, level):
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
    slide = usi.open_image(wsi)
    if level < 0:
        level = slide.level_count + level
    name_wsi, ext = os.path.splitext(os.path.basename(wsi))
    new_path = name_wsi + '.jpg'
    WSI = usi.get_whole_image(slide=slide, level=level, numpy=False)
    WSI = WSI.convert("RGB")
    WSI.save(new_path)

parser = ArgumentParser()
parser.add_argument('--path', required=True, type=str, help="path of the files to downsample (tiff or svs files)")
parser.add_argument('--level', type=int, default = 2, help="scale to which downsample. I.e a scale of 2 means dimensions divided by 2^2")
args = parser.parse_args()

downsize(args.path, args.level)
