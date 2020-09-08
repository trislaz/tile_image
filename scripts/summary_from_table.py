from argparse import ArgumentParser
from summary_fig import main
import os
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--table', type=str, help='path to the table with the ID of the slides')
parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
parser.add_argument('--embed_path', type=str, help='path to the embeddings')
parser.add_argument('--raw_path', type=str, help='path to the raw images (ndpi, svs... )')
parser.add_argument('--out', type=str, help='name of the folder where storing the outputs', default='visus')
args = parser.parse_args()

df = pd.read_csv(args.table)
IDs = df['ID'].values
for i in IDs:
    main(model_path=args.model_path, wsi_ID=i, embed_path=args.embed_path,
        raw_path=args.raw_path, out=args.out, table=args.table)
