from argparse import ArgumentParser
import torch

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--path_wsi', required=True, type=str, help="path of the files to downsample (tiff or svs files)")
    parser.add_argument('--path_mask', type=str, default='no', help='either a path to the xml file, if no, then the whole image is tiled')
    parser.add_argument('--level', type=int, default = 1, help="scale to which downsample. I.e a scale of 2 means dimensions divided by 2^2")
    parser.add_argument('--size', type=int, default = 256, help="size of patches")
    parser.add_argument('--auto_mask', type=int, default=1, help="if 1, mask is .npy, .xml else")
    parser.add_argument('--tiler', type=str, default='simple', help='type of tiler : imagenet | simple | simclr')
    parser.add_argument('--path_outputs', type=str, help='output folder path', default='.')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args