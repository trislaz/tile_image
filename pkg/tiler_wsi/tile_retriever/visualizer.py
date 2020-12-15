from deepmil.predict import load_model
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, distance_transform_bf
import matplotlib.patches as patches
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
from glob import glob
import yaml
import useful_wsi as usi
from copy import copy
import numpy as np
import pickle
import pandas as pd
from model_hooker import HookerMIL
import os 

def make_background_neutral(heatmap):
    """
    For the sake of visibility, puts the background neutral.
     = puts the background at the center of the heatmap distri.
    """
    heatmap_neutral = copy(heatmap)
    heatmap_neutral[heatmap_neutral == 0] = np.mean([heatmap_neutral.max(), heatmap_neutral.min()])
    return heatmap_neutral


class VisualizerMIL:
    """
    Class that allows the hook and storage of MIL features during a forward pass.
    Need to be initializerd with the path of a model
    (if during the whole analysis, global raw path and embedding path wont change, they can be passed here for initialization).
    Then a forward_pass can be called on a specific wsi_ID. 
    VisualizerMIL.hooker stores :
        * .tiles_weights
        * .reprewsi
        * .head_average
    When calling several times the forward pass, some of the key features, useful to compute
    dataset-level visualisation (umap etc...) are stored in the .stored dict.
    Also compute plots
    """
    def __init__(self, model, path_raw=None, path_emb=None, table=None, level_visu=-1):
        self.k = k
        ## Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model, self.device)

        # Par défaut on considère le dataset d'entrainement 
        # = données contenues dans args.
        if table is None:
            table = self.model.args["table_data"]
        if path_embed is None:
            path_embed = self.model.args["wsi"]
        if path_raw is None:
            path_raw = self.model.args["raw_path"]
        self.table = table
        self.path_embed = path_embed
        self.path_raw = path_raw

        ## Model parameters
        self.num_heads = self.model.args.num_heads
        self.model_name = self.model.args.model_name
        self.target_name = self.model.args.target_name
        self.level_visu = level_visu

        ## Instanciate hooker
        self.hooker = HookerMIL(self.model.network, self.num_heads)
        
        ## list that have to be filled with various WSI
        self.stored = {"names":[], "heads":[], "reprewsi":[], 'label':[]}

        self.gt = None
        self.pred = None
        self.wsi_ID = None

    def forward_pass(self, wsi_ID: str, path_emb=None) -> str:
        """forward_pass.
    
        Makes a forward pass of wsi_ID, hooks the interesting layers
        in the hooker. Gets also the info-object.
    
        :param wsi_ID: name of the wsi, without the extension
        :type wsi_ID: str
        :param path_emb: path where are stored the outputs folders of the tiling pipeline (info, mat_pca etc..).
        :type path_emb: str
        """
        if path_emb is None:
            path_emb = self.path_emb
        self.wsi_ID = wsi_ID 
        self.info = self._get_info(wsi_ID, path_emb)
        wsi_embedded_path = os.path.join(embeddings, 'mat_pca', wsi_ID + "_embedded.npy")
        input_wsi = self._preprocess(wsi_embedded_path)
        _, pred = self.model.predict(input_wsi)
        self.gt = self._extract_groundtruth(table=self.table, wsi_ID=wsi_ID)
        self.pred = pred
        self.store_wsi_features()
        return pred

    def make_visu(self, figtypes:str, wsi_ID:str, path_emb:str, path_raw:str, head=1):
        pred = self.forward_pass(wsi_ID, path_emb)
        figs = {}
        for ft in figtypes:
            figs[ft] = getattr(self, 'create_{}_fig'.format(ft))(path_raw, pred)
        return figs

    def store_wsi_features(self):
        self.stored["names"].append(self.wsi_ID)
        self.stored['heads'].append(self.hooker.head_average)
        self.stored['reprewsi'].append(self.hooker.reprewsi)
        self.stored['label'].append(self.gt)

    def flush_wsi_features(self):
        self.stored = {"names":[], "heads":[], "reprewsi":[], "label":[]}

    def create_heatmap_fig(self, path_raw=None):
        if path_raw is None:
            self.path_raw = path_raw
        heatmaps = self._get_heatmaps()
        num_heads = self.num_heads
        num_cases = round((num_heads + 1)/2)+1
        gridsize = (num_cases, 2)
        fig = plt.figure(figsize=(num_cases * 10, 20))
        for nh in range(num_heads):
            r = nh // 2
            c = nh % 2
            heatarray = self.make_background_neutral(heatmaps[:,nh])
            heatmap = plt.subplot2grid(gridsize, (r, c), rowspan=1, colspan=1, fig=fig)
            hm = heatmap.imshow(heatarray, cmap='coolwarm', vmin=heatarray.min())
            heatmap.set_title('Scores')
            heatmap.set_axis_off()
        visu = plt.subplot2grid(gridsize, (int(((num_heads+1)/2)//2)+1, int(((num_heads+1)/2)%2)), rowspan=1, colspan=1, fig=fig)
        visu.imshow(self._get_down_image(os.path.join(path_raw, wsi_ID)))
        visu.set_axis_off()
        add_titlebox(visu, self._make_message())
        fig.tight_layout()
        return fig

    def create_summary_fig(self, path_raw=None, head=0):
        if path_raw is None:
            path_raw = self.path_raw
        color_tile = {0:'red', 1:'blue'}
        legend_elements = [patches.Patch(facecolor=color_tile[0], edgecolor='black', label='Highest scores'), 
                          patches.Patch(facecolor=color_tile[1], edgecolor='black', label='Lowest scores')]

        ## Making the heatmaps
        heatmaps = self._get_heatmaps()
        heatmap = heatmap[:,head]

        assert self.images['topk'] is not None, "You have to first self.get_images"
        gridsize = (6, 4)
        fig = plt.figure(figsize=(30, 20))
        visu = plt.subplot2grid(gridsize, (2, 0), rowspan=2, colspan=2, fig=fig)
        visu.imshow(self.images['wsi_down'])
        visu.set_axis_off()
        hm_ax = plt.subplot2grid(gridsize, (2, 2), rowspan=2, colspan=2, fig=fig)
        hm = hm_ax.imshow(self.make_background_neutral(heatmap), cmap='coolwarm')
        heatmap.set_title('Scores')
        fig.colorbar(hm, ax=hm_ax)
        heatmap.set_axis_off()

        # Best/worst tiles
        topk = {}
        topk['bestk'], topk['lowk'] = self._get_topk_tiles()
        tiles = []
        for l in ['bestk', 'lowk']:
            for c in range(4):
                ax = plt.subplot2grid(gridsize, (l,c), fig=fig)
                ind = topk[l][c]
                im = self._get_image(path_raw, ind)
                ax.imshow(im)
                ax = set_axes_color(ax, color=color_tile[l]) 
                tiles.append(ax)
                visu = self._plot_loc_tile(visu, color=color_tile[l], para=self.info['paradict'][c])
        visu.legend(handles=legend_elements, loc='upper right', fontsize=12, handlelength=2)
        add_titlebox(visu, self._make_message(self.pred))
        fig.tight_layout()
        return fig

    def _get_topk_tiles(self, heatmap, k=4):
        infomat = self.info['infomat'].flatten()
        mask = infomat > -1
        heatmap[mask].flatten()
        heatmap = heatmap.flatten()[mask]
        indices = np.argsort(heatmap)
        topk = indices[-k:]
        lowk = indices[:k]
        topk_i = infomat[mask][topk]
        lowk_i = infomat[mask][lowk]
        return topk, lowk

     def _get_down_image(self, wsi):
        """get the downsampled image (numpy format) at the desired downsampling factor.
        """
        self.wsi = usi.open_image(wsi)
        if self.level_visu < 0:
            self.level_visu = self.wsi.level_count + self.level_visu
        image = usi.get_whole_image(self.wsi, level=self.level_visu, numpy=True)
        return image
        
    def _preprocess(self, input_path):
        """preprocess the input to feed the model
    
        Args:
            input_path (str): str to the input path 
        """
        depth = self.model.args.feature_depth
        inp = np.load(input_path)[:,:depth]
        inp = torch.Tensor(inp)
        inp = inp.unsqueeze(0)
        inp = inp.to(self.device)
        return inp

    def _get_heatmap(self):
        assert self.wsi_ID, "You need to forward pass a WSI before"
        tiles_weights =  self.hooker.tiles_weights
        infomat = self.info['infomat']
        size = (infomat.shape[0], infomat.shape[1], self.num_heads)
        infomat = infomat.flatten()
        heatmaps = np.zeros((len(infomat), self.num_heads))
        for o, i in enumerate(infomat):
            if i > 0:
                for nh in range(self.num_heads):
                    heatmap[o, :] = self.tiles_weights[int(i), :]
        heatmaps = heatmap.reshape(size)
        return heatmaps

    def _get_info(self, wsi_ID, path_emb):
        wsi_info_path = os.path.join(path_emb, 'info')
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy')
        infomat = np.load(infomat)
        infomat = infomat.T 
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
        return {'infomat':infomat, 'paradict': infodict , 'paralist': self._infodict_to_list(infodict)}

    def _get_image(self, raw_path, indice):
        para = self.info['paratiles'][indice]
        image = usi.get_image(slide=os.path.join(self.raw_path, self.wsi_ID), 
                para=paras, numpy=True)
        return image

   def _plot_loc_tile(self, ax, color, para):
        args_patch = {'color':color, 'fill': False, 'lw': 5}
        top_left_x, top_left_y = usi.get_x_y_from_0(self.wsi, (para['x'], para['y']), self.level_visu)
        width, height = usi.get_size(self.wsi, (para['xsize'], para['ysize']), para['level'], self.level_visu)
        plot_seed = (top_left_x, top_left_y)
        patch = patches.Rectangle(plot_seed, width, height, **args_patch)
        ax.add_patch(patch)
  
     def _infodict_to_list(dictio):
        """Interface to use ùseful_wsi`that needs a list for parameter
        Because this is how have been implemented the tiling by peter

        Args:
            dictio (dict): my parameter dict (x, y, xsize, ysize, level)

        Returns:
            list: list of parameters, in the good order.
        """
        para = []
        para.append(dictio['x'])
        para.append(dictio['y'])
        para.append(dictio['xsize'])
        para.append(dictio['ysize'])
        para.append(dictio['level'])
        return para

    def _make_message(self,pred):
        if self.result_pred is '':
            msg = "Prediction of {} : {}.".format(self.target_name, pred)
        else:
            msg = "Prediction of {} : {}. Ground truth: {}".format(self.target_name, pred, self.gt)
        return msg

    def _extract_groundtruth(self, table, wsi_ID):
        if isinstance(table, str):
            if os.path.isfile(table):
                table = pd.read_csv(table)
            else:
                return "No table"
        if wsi_ID in table['ID'].values:
            gt = table[table['ID'] == wsi_ID][self.target_name].values[0]
            gt = self.model.target_correspondance[gt]
        else:
            return "not in table"
        return gt
 
