from argparse import ArgumentParser
from summary_fig import main as process_slide
import os
import torch
import pandas as pd

def main(model_path):
    state = torch.load(model_path, map_location='cpu')
    raw_path = state['args'].raw_path #"/mnt/data4/gbataillon/Dataset/Curie/Global/raw"
    embed_path = state['args'].wsi#"/mnt/data4/tlazard/data/curie/curie_recolo_tiled/imagenet/size_256/res_2" #
    table = state['args'].table_data #"/mnt/data4/tlazard/3LST100.csv" #
    test = state['args'].test_fold
    out = os.path.join(os.path.dirname(model_path), 'summaries_test_{}'.format(test))
    df = pd.read_csv(table)
    test_rows = df['test'] == test
    IDs = df[test_rows]['ID'].values
    for i in IDs:
        process_slide(model_path=model_path, wsi_ID=i, embed_path=embed_path,
            raw_path=raw_path, out=out, table=table)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    args = parser.parse_args()
    
    main(model_path=args.model_path)

