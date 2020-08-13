from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from tiler_wsi.tile_retriever.tile_sampler import MILHeatmat

def main(model_path, wsi_ID, embed_path, raw_path, out, table, store_best):
    mhm = MILHeatmat(model_path)    
    mhm.get_images(wsi_ID=wsi_ID, embeddings=embed_path, raw=raw_path, table=table)
    fig = mhm.get_summary_fig()
    out = os.path.join('.', out, mhm.result_pred)
    out_path = os.path.join(out, wsi_ID+'_summary.jpg')
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    parser.add_argument('--embed_path', type=str, help='path to the embeddings')
    parser.add_argument('--raw_path', type=str, help='path to the raw images (ndpi, svs... )')
    parser.add_argument('--wsi_ID', type=str, help='ID = name of the slide to predict')
    parser.add_argument('--out', type=str, help='folder to store the output', defautl='.')
    parser.add_argument('--table', type=str, help='path to the table data containing ground truth. Same format as the table data\
                        used for training. If None, no outcome is given for the prediction (success or failure)', default=None)
    args = parser.parse_args()

    main(args.model_path, args.wsi_ID, args.embed_path, args.raw_path, args.out, args.table)

