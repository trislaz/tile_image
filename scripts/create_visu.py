from argparse import ArgumentParser
from summary_fig import main as process_slide
from tiler_wsi.tile_retriever.visualizer import VisualizerMIL
from tiler_wsi.tile_retriever.best_representative_tile import TileRepresentant
import matplotlib.pyplot as plt
import shutil 
import umap
import os
import torch
import numpy as np
import pandas as pd

def main(model_path:str, summary:bool, heatmaps:bool, reprewsi:bool):
    visu = VisualizerMIL(model=model_path)
    out = os.path.join(os.path.dirname(model_path),
            'summaries_{}'.format(os.path.splitext(os.path.basename(model_path))))
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(visu.table)
    IDs = df['ID'].values
    for i in IDs:
        visu.forward_pass(i)
        if summary:
            fig = visu.create_summary_fig()
            fig.savefig(os.path.join(out, 'summary'+i+'.pdf'))
        if heatmaps:
            fig = visu.create_heatmap_fig()
            fig.savefig(os.path.join(out, 'heatmaps'+i+'.pdf'))

    ## Créer les tiles_representatives + umap:
    if reprewsi:
        tile_repre = TileRepresentant(visu.path_embed, visu.path_raw)
        names = visu.stored['names']
        heads = visu.stored['heads']
        reprewsi = visu.stored['reprewsi']
        labels = visu.stored['label']
        out_r = os.path.join(out, 'representants')
        os.makedirs(out_r, exist_ok=True)
        for o, h in enumerate(heads):
            average = h[0, :]
            image = tile_repre.get_image(average)
            image.save(os.path.join(out_r, names[o]+'.jpg'))
        np.save(os.path.join(out, 'names.npy'), np.array(names))
        np.save(os.path.join(out, 'repre_wsi.npy'), np.array(reprewsi))
        np.save(os.path.join(out, 'labels.npy'), np.array(labels))

        # umap transform
        umap = umap.UMAP(n_neighbors=45)
        projection = umap.fit_transform(reprewsi)
        np.save(os.path.join(out, 'projection_coordinates.npy'), projection)
        shutil.copy('~/Documents/cbio/projets/tile_image/scripts/make_dynamic_umap.py', 
                os.path.join(out, 'make_dynamic_umap.py'))
        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--heatmaps', action='store_true')
    parser.add_argument('--reprewsi', action='store_true')
    args = parser.parse_args()
    
    main(model_path=args.model_path,
            summary=args.summary,
            heatmaps=args.heatmap, 
            reprewsi=args.reprewsi)
    


