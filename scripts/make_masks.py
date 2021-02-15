from glob import glob
from tiler_wsi.tile_image.auto_cc_masks import main
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the raw images -all format accepted-')
parser.add_argument('--level_mask', type=int, help='level to which is extracted the mask', default=-1)
parser.add_argument('--out_path', type=str, help='path of the outputed masks', default='.')
args = parser.parse_args()

files = glob(os.path.join(args.path, '*.*'))
for f in files:
    main(f, args.out_path, args.level_mask)
