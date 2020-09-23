from argparse import ArgumentParser
from summary_fig import main
import os
import torch
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
args = parser.parse_args()

state = torch.load(args.model_path, map_location='cpu')
raw_path = state['args'].raw_path
embed_path = state['args'].wsi
table = state['args'].table_data
test = state['args'].test_fold
out = os.path.join(os.path.dirname(args.model_path), 'summaries_test_{}'.format(test))
df = pd.read_csv(table)
test_rows = df['test'] == test
IDs = df[test_rows]['ID'].values
for i in IDs:
    main(model_path=args.model_path, wsi_ID=i, embed_path=embed_path,
        raw_path=raw_path, out=out, table=table)
