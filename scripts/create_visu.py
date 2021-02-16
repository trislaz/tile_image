from argparse import ArgumentParser
from tiler_wsi.tile_retriever.visualizer import VisualizerMIL
from tiler_wsi.tile_retriever.best_representative_tile import TileRepresentant
import matplotlib.pyplot as plt
from glob import glob
import shutil 
import umap
import os
import torch
import numpy as np
import pandas as pd

def main(model_path:str, summary:bool, heatmaps:bool, heatmap_target, reprewsi:bool, toptile:bool, allslides:bool, make_mask:bool):

    visu = VisualizerMIL(model=model_path)
    out = os.path.join(os.path.dirname(model_path),
            'summaries_{}'.format(os.path.basename(model_path).replace('.pt.tar', '')))
    os.makedirs(out, exist_ok=True)
    if toptile:
        os.makedirs(os.path.join(out, 'best_tiles'), exist_ok=True)
    if make_mask:
        os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    if heatmaps:
        os.makedirs(os.path.join(out, 'heatmap'), exist_ok=True)
    if heatmap_target:
        os.makedirs(os.path.join(out, 'hm_target'), exist_ok=True)
    if summary:
        os.makedirs(os.path.join(out, 'summary'), exist_ok=True)


    if allslides:
        IDs = glob(os.path.join(visu.path_emb,'mat_pca','*.npy'))
        IDs = [os.path.basename(x).replace('_embedded.npy', '') for x in IDs]
    else:
        df = visu.table
        IDs = df[df['test'] == int(visu.model.args.test_fold)]['ID'].values
    for i in IDs:
        if ',' in i:
            continue
#        try:
        visu.forward(i)
 #       except:
 #           print('galere sur {}'.format(i))
 #           continue
        if toptile:
            image = visu.get_best_tile()
            image.save(os.path.join(out, 'best_tiles',str(visu.pred)+'_'+i+'.png'))
        if summary:
            fig = visu.create_summary_fig()
            fig.savefig(os.path.join(out, 'summary',i+'.pdf'))
            plt.close('all')
        if heatmaps:
            fig = visu.create_heatmap_fig()
            fig.savefig(os.path.join(out, 'heatmap',i+'.jpg'))
            plt.close('all')
        if make_mask:
            mask = visu.create_masks(N=500)
            np.save(os.path.join(out, 'masks', i+'.npy'), mask)
        if heatmap_target:
            fig = visu.create_heatmap_target_fig()
            fig.savefig(os.path.join(out, 'hm_target', i+'.jpg'))

    ## Créer les tiles_representatives + umap:
    if reprewsi:
        tile_repre = TileRepresentant(visu.path_emb, visu.path_raw, N=1000)
        names = visu.stored['names']
        heads = visu.stored['heads']
        reprewsi = visu.stored['reprewsi']
        labels = visu.stored['label']
        preds = visu.stored['pred']
        out_r = os.path.join(out, 'representants')
        os.makedirs(out_r, exist_ok=True)
        for o, h in enumerate(heads):
            average = h[0, :]
            image = tile_repre.get_image(average)
            image.save(os.path.join(out_r, str(preds[o])+'_'+names[o]+'.png'))
        np.save(os.path.join(out, 'names.npy'), np.array(names))
        np.save(os.path.join(out, 'repre_wsi.npy'), np.array(reprewsi))
        np.save(os.path.join(out, 'labels.npy'), np.array(labels))
        np.save(os.path.join(out, 'heads.npy'), heads)
        np.save(os.path.join(out, 'preds.npy'), preds)

        # umap transform
        um = umap.UMAP(n_neighbors=15)
        projection_h = um.fit_transform(np.vstack(np.array(heads)))#um.fit_transform(np.vstack(np.array(reprewsi)))
        projection_r = um.fit_transform(np.vstack(np.array(reprewsi)))
        np.save(os.path.join(out, 'projection_head.npy'), projection_h)
        np.save(os.path.join(out, 'projection_reprewsi.npy'), projection_r)
        shutil.copy('/gpfsdswork/projects/rech/gdg/uub32zv/packages/tile_image/scripts/make_dynamic_umap.py', 
                os.path.join(out, 'make_dynamic_umap.py'))
        
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--heatmaps', action='store_true')
    parser.add_argument('--reprewsi', action='store_true')
    parser.add_argument('--toptile', action='store_true')
    parser.add_argument('--allslides', action='store_true')
    parser.add_argument('--make_mask', action='store_true')
    parser.add_argument('--heatmap_target', action='store_true')
    args = parser.parse_args()
    
    main(model_path=args.model_path,
            summary=args.summary,
            heatmaps=args.heatmaps,
            heatmap_target=args.heatmap_target,
            toptile=args.toptile,
            allslides=args.allslides,
            make_mask=args.make_mask,
            reprewsi=args.reprewsi)
    


