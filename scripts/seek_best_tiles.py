from tiler_wsi.tile_retriever.visualizer import TileSeeker
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--n_best', type=int, default=20)
parser.add_argument('--min_prob', action='store_true')
args = parser.parse_args()

ts = TileSeeker(model=args.model, n_best=args.n_best, min_prob=args.min_prob)
ts.forward_all()
ts.extract_images()
for k in ts.store_image:
    os.makedirs(k, exist_ok=True)
    [x.save(os.path.join(k, 'tile_{}.jpg'.format(o))) for o,x in enumerate(ts.store_image[k])]


