#%%
#from .tile_functionnal import compute_distance
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
import numpy as np
import pickle
import pandas as pd
import os 

def add_titlebox(ax, text):
    ax.text(.05, .05, text,
        horizontalalignment='left',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        fontsize=25)
    return ax

def set_axes_color(ax, color='orange'):
    dirs = ['bottom', 'top', 'left', 'right']
    args = {x:False for x in dirs}
    args.update({'label'+x:False for x in dirs})
    args.update({'axis':'both', 'which': 'both'})
    ax.tick_params(**args)
    for sp in ax.spines:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(5)
    return ax

def filename(path:str):
    return os.path.splitext(os.path.basename(path))[0].split('_embedded')[0]

## For the moment, I have to transpose infomat to have the good representation.
## Change directly in the tiling process, then here
class MILHeatmat:
    def __init__(self, model, level_visu=-1, k=5):
        self.k = k
        self.model = load_model(model)
        self.target_name = self.model.args.target_name
        self.level_visu = level_visu
        self.tiles_weights = None
        self._get_hook()
        self.gt = None
        self.pred = None
        self.result_pred = ''
        self.infomat = None
        self.wsi_ID = None
        self.wsi_embedded_path = None
        self.wsi_raw_path = None
        self.params_lowk = None
        self.params_topk = None
        ## Attributes that can be plotted:
        self.images = {'heatmap':None,
                        'topk':None,
                        'lowk': None,
                        'wsi_down':None}
        self.scores = {'topk': None, 
                        'lowk':None}
        self.params = {'topk':None, 
                        'lowk': None}

    def get_images(self, wsi_ID, embeddings, raw, table=None):
        """Generates data from the prediction of a model and stores it in dict attributes.
            * self.images with keys heatmap | topk | lowk | wsi_down (all in npy format)
            * self.scores with keys topk | lowk are the scores of the topk tiles and worst k tiles
            * self.params with keys topk | lowk are the params of the top and worst tiles (in the 0 level) 

        Args:
            wsi_ID (str): name of the image on which to use the model. WSI and embeddings must be named after it.
            embeddings (str): out path of a tile-image process; where are stored the embeddings and their info.
            raw (str): path of the raw WSI images.
            table (str, optional): Either a path to a table data. If no, then no info on the label is available.
                Defaults to 'no'.
        """
        self.wsi_ID = wsi_ID
        self.wsi_raw_path = glob(os.path.join(raw, wsi_ID + ".*"))[0] # implemented only for npy embedings.
        wsi_embedded_path = os.path.join(embeddings, 'mat_pca', wsi_ID + "_embedded.npy")
        wsi_info_path = os.path.join(embeddings, 'info')
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy')
        infomat = np.load(infomat)
        infomat = infomat.T # TODO A enlever plus tard
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
        #forward
        input_wsi = self._preprocess(wsi_embedded_path)
        _, pred = self.model.predict(input_wsi)
        # transform hooks to infomat.
        # Fills the images dict
        heatmap = self._transfer_to_infomat(infomat)
        wsi_down = self._get_down_image(self.wsi_raw_path)
        self._set_best_tiles(self.wsi_raw_path, heatmap, self.k, infomat, infodict)
        self.images['heatmap'] = heatmap
        self.images['wsi_down'] = wsi_down

        self.pred = self.model.args.target_correspondance[int(pred)]
        if table is not None:
            self.gt = self._extract_groundtruth(table, wsi_ID)
            self.result_pred = 'success' if self.gt == self.pred else 'failure'

    def _extract_groundtruth(self, table, wsi_ID):
        if isinstance(table, str):
            table = pd.read_csv(table)
        gt = table[table['ID'] == wsi_ID][self.target_name].values[0]
        return gt
        
    def _plot_loc_tile(self, ax, color, para):
        args_patch = {'color':color, 'fill': False, 'lw': 5}
        top_left_x, top_left_y = usi.get_x_y_from_0(self.wsi, (para['x'], para['y']), self.level_visu)
        width, height = usi.get_size(self.wsi, (para['xsize'], para['ysize']), para['level'], self.level_visu)
        plot_seed = (top_left_x, top_left_y)
        patch = patches.Rectangle(plot_seed, width, height, **args_patch)
        ax.add_patch(patch)
        return ax

    def _set_best_tiles(self, wsi_raw_path, heatmap, k, infomat, infodict):
        mask = infomat > 0
        mask = mask.flatten()
        heatmap = heatmap.flatten()[mask]
        infomat = infomat.flatten()[mask]
        assert heatmap.shape == infomat.shape, "Infomat and heatmap do not have the same shape"
        indices = np.argsort(heatmap)
        topk = indices[-k:]
        lowk = indices[:k]
        topk_i = infomat[topk]
        lowk_i = infomat[lowk]
        self.scores['topk'] = heatmap[topk]
        self.scores['lowk'] = heatmap[lowk]
        self.params['topk'] = [infodict[i] for i in topk_i]
        self.params['lowk'] = [infodict[i] for i in lowk_i]
        self.images['topk'] = [usi.get_image(wsi_raw_path, self._infodict_to_list(infodict[i]), numpy=True) for i in topk_i]
        self.images['lowk'] = [usi.get_image(wsi_raw_path, self._infodict_to_list(infodict[i]), numpy=True) for i in lowk_i]

    @staticmethod
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
    
    def get_summary_fig(self):
        color_tile = {0:'red', 1:'blue'}
        ref_l = {0: 'topk', 1: 'lowk'}
        legend_elements = [patches.Patch(facecolor=color_tile[0], edgecolor='black', label='Highest scores'), 
                          patches.Patch(facecolor=color_tile[1], edgecolor='black', label='Lowest scores')]
        assert self.images['topk'] is not None, "You have to first self.get_images"
        gridsize = (6, 4)
        fig = plt.figure(figsize=(20, 20))
        visu = plt.subplot2grid(gridsize, (2, 0), rowspan=2, colspan=2, fig=fig)
        visu.imshow(self.images['wsi_down'])
        visu.set_axis_off()
        heatmap = plt.subplot2grid(gridsize, (2, 2), rowspan=2, colspan=2, fig=fig)
        hm = heatmap.imshow(self.images['heatmap'], cmap='coolwarm', vmin=self.scores['lowk'][0])
        fig.colorbar(hm, ax=heatmap)
        heatmap.set_axis_off()
        tiles = []
        for l in range(2):
            for c in range(4):
                ax = plt.subplot2grid(gridsize, (l,c), fig=fig)
                im = self.images[ref_l[l]][c]
                ax.imshow(im)
                ax = set_axes_color(ax, color=color_tile[l]) 
                ax.set_title('score: {}'.format(self.scores[ref_l[l]][c]))
                tiles.append(ax)
                visu = self._plot_loc_tile(visu, color=color_tile[l], para=self.params[ref_l[l]][c])
        visu.legend(handles=legend_elements, loc='upper right', fontsize=12, handlelength=2)
        add_titlebox(visu, self._make_message())
        fig.tight_layout()
        return fig

    def _make_message(self):
        if self.result_pred is '':
            msg = "Prediction of {} : {}.".format(self.target_name, self.pred)
        else:
            msg = "Prediction of {} : {}. Ground truth: {}".format(self.target_name, self.pred, self.gt)
        return msg

    def compute_and_save(self, wsi_ID, embeddings, raw):
        out_path = os.path.join(self.out_path, wsi_ID + '_heatmap.jpg')
        fig, _ = self.compute_heatmap(wsi_ID, embeddings, raw)
        fig.savefig(out_path)

    def _transfer_to_infomat(self, infomat):
        """transfers the weights hooked on to the infomat.

        Args:
            infomat (npy): npy array containing the index of each tile
        """
        assert self.tiles_weights is not None, "you have to compute one forward pass of the model"
        size = infomat.shape
        infomat = infomat.flatten()
        heatmap = np.zeros(len(infomat))
        for o, i in enumerate(infomat):
            if i > 0:
                heatmap[o] = self.tiles_weights[int(i)]
        heatmap = heatmap.reshape(size)
        return heatmap

    ## for debugging
    def plot_infomat(self, image, infomat):
        figure = plt.figure(figsize=(25, 20))
        plt.imshow(np.mean(image, axis=2), cmap='gray', interpolation='bilinear')
        plt.axis('off')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.imshow(infomat, cmap='red', interpolation='nearest', alpha=.5, extent=(xmin,xmax,ymin,ymax))   # for heatmap to overlap
        return figure

    def plot_heatmap(self, image, heatmap):
        #figure = plt.figure(figsize=(20, 20))
        #plt.imshow(np.mean(image, axis=2), cmap='gray', interpolation='bilinear')
        #plt.axis('off')
        #xmin, xmax = plt.xlim()
        #ymin, ymax = plt.ylim()
        #plt.imshow(heatmap.T, cmap='coolwarm', interpolation='nearest', alpha=.6 extent=(xmin,xmax,ymin,ymax))   # for heatmap to overlap
        plt.figure(figsize=(20, 20))
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(heatmap.T, cmap='red')
        return figure

    def _get_hook(self):
        def hook_tiles(m, i, o):
            tiles_weights = o
            self.tiles_weights = tiles_weights.squeeze().detach().cpu().numpy()

        def get_all_layers(net):
            for name, layer in net.named_children():
                if list(layer.children()):
                    get_all_layers(layer)
                if name == 'weight_extractor':
                    hook_layer = list(layer.children())[2]
                    hook_layer.register_forward_hook(hook_tiles)
                    print('Hook in place, captain')
        get_all_layers(self.model.network)

    def _preprocess(self, input_path):
        """preprocess the input to feed the model

        Args:
            input_path (str): str to the input path 
        """
        depth = self.model.args.feature_depth
        inp = np.load(input_path)[:,:depth]
        inp = torch.Tensor(inp)
        inp = inp.unsqueeze(0)
        inp = inp.to(self.model.device)
        return inp

    def _get_down_image(self, wsi):
        """get the downsampled image (numpy format) at the desired downsampling factor.
        """
        self.wsi = usi.open_image(wsi)
        if self.level_visu < 0:
            self.level_visu = self.wsi.level_count + self.level_visu
        image = usi.get_whole_image(self.wsi, level=self.level_visu, numpy=True)
        return image
        
if __name__ == "__main__":

#%%
    model = '/Users/trislaz/Documents/cbio/projets/tile_image/data_test/model.pt.tar'
    wsi_ID = '302073T'
    embeddings = '/Users/trislaz/Documents/cbio/projets/tile_image/data_test/embeddings'
    raw = '/Users/trislaz/Documents/cbio/projets/tile_image/data_test'
    hm = MILHeatmat(model=model, level_visu=-1)
    hm.get_images(wsi_ID, embeddings, raw)
    fig = hm.get_summary_fig()


