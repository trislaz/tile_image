from argparse import ArgumentParser
import os
from tiler_wsi.tile_retriever.tile_sampler import MILHeatmat

def main(model_path, wsi_ID, embed_path, raw_path, ext_wsi, out_path):
    mhm = MILHeatmat(model_path, ext_wsi=ext_wsi)    
    mhm.get_images(wsi_ID=wsi_ID, embeddings=embed_path, raw=raw_path)
    fig = mhm.get_summary_fig()
    fig.savefig(os.path.join(out_path, wsi_ID+'_summary.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    parser.add_argument('--embed_path', type=str, help='path to the embeddings')
    parser.add_argument('--raw_path', type=str, help='path to the raw images (ndpi, svs... )')
    parser.add_argument('--wsi_ID', type=str, help='ID = name of the slide to predict')
    parser.add_argument('--out_path', type=str, help='folder to store the output', defautl='.')
    parser.add_argument('--ext_wsi', type=str, help='extension of the WSIs', default='ndpi')
    args = parser.parse_args()
    main(args.model_path, args.wsi_ID, args.embed_path, args.raw_path, args.ext_wsi, args.out_path)

