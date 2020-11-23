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
    if 'test_fold' in vars(state['args']):
        test = None#state['args'].test_fold
    else:
        test = None
    out = os.path.join(os.path.dirname(model_path), 'summaries_test_{}'.format(test))
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(table)
    if test:
        test_rows = df['test'] == test
    else:
        test_rows = df.columns
    IDs = df[test_rows]['ID'].values
    for i in IDs:
        try:
            process_slide(model_path=model_path, wsi_ID=i, embed_path=embed_path,
                raw_path=raw_path, out=out, table=table)
        except:
            continue

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    args = parser.parse_args()
    
    main(model_path=args.model_path)

