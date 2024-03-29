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
from copy import copy
import numpy as np
import pickle
import pandas as pd
import os 

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)[0]
        self.probs = F.softmax(self.logits, dim=0)
        return self.probs.sort(dim=0, descending=True), torch.argmax(self.logits).item()  # ordered results

    def backward(self):
        """
        Class-specific backpropagation
        """
        #one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        index = torch.argmax(self.logits).item()
        self.logits[index].backward(retain_graph=True)

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam.squeeze().numpy()

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model, self.device)
        self.num_heads = self.model.args.num_heads
        self.model_name = self.model.args.model_name
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
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy') infomat = np.load(infomat)
        infomat = infomat.T # TODO A enlever plus tard
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
        #forward
        input_wsi = self._preprocess(wsi_embedded_path)
        print(input_wsi.shape)
        _, pred = self.model.predict(input_wsi)
        # transform hooks to infomat.
        # Fills the images dict
        heatmap = self._transfer_to_infomat(infomat)

        ###For debugging and interpretation of scores:
        #heatmap_norm = self._transfert_feature_norm_to_infomat(infomat)
        #self.images['heatmap_norm'] = heatmap_norm

        wsi_down = self._get_down_image(self.wsi_raw_path)
        #self._set_best_tiles(self.wsi_raw_path, heatmap, self.k, infomat, infodict)
        self.images['heatmap'] = heatmap
        self.images['wsi_down'] = wsi_down
        self.pred = pred#self.model.args.target_correspondance[int(pred)]
        if table is not None:
            self.gt = self._extract_groundtruth(table, wsi_ID)
            self.result_pred = 'success' if self.gt == self.pred else 'failure'

    def _extract_groundtruth(self, table, wsi_ID):
        if isinstance(table, str):
            table = pd.read_csv(table)
        gt = table[table['ID'] == wsi_ID][self.target_name].values[0]
        gt = self.model.target_correspondance[gt]
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
        fig = plt.figure(figsize=(30, 20))
        visu = plt.subplot2grid(gridsize, (2, 0), rowspan=2, colspan=2, fig=fig)
        visu.imshow(self.images['wsi_down'])
        visu.set_axis_off()
        heatmap = plt.subplot2grid(gridsize, (2, 2), rowspan=2, colspan=2, fig=fig)
        hm = heatmap.imshow(self._make_background_neutral(self.images['heatmap']), cmap='coolwarm', vmin=self.scores['lowk'][0])
        heatmap.set_title('Scores')
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
        add_titlebox(visu, self._make_message(self.pred))
        fig.tight_layout()
        return fig

    def get_heatmaps_fig(self): 
        heatmaps = self.images['heatmap']
        num_heads = heatmaps.shape[-1]
        num_cases = round((num_heads + 1)/2)+1
        gridsize = (num_cases, 2)
        fig = plt.figure(figsize=(num_cases * 10, 20))
        for nh in range(num_heads):
            r = nh // 2
            c = nh % 2
            heatarray = self._make_background_neutral(self.images['heatmap'][:,:,nh])
            heatmap = plt.subplot2grid(gridsize, (r, c), rowspan=1, colspan=1, fig=fig)
            hm = heatmap.imshow(heatarray, cmap='coolwarm', vmin=heatarray.min())
            heatmap.set_title('Scores')
            heatmap.set_axis_off()
        visu = plt.subplot2grid(gridsize, (int(((num_heads+1)/2)//2)+1, int(((num_heads+1)/2)%2)), rowspan=1, colspan=1, fig=fig)
        visu.imshow(self.images['wsi_down'])
        visu.set_axis_off()
        add_titlebox(visu, self._make_message())
        fig.tight_layout()
        return fig

    def get_summary_fig_norms(self):
        fig = plt.figure(figsize=(20, 20))
        gridsize = (4, 4)
        h1, h2 = self.images['heatmap'], self.images['heatmap_norm']
        heatmap = np.sqrt(np.multiply(h1, h2))
        heatmap[heatmap==0] = np.mean([heatmap.max(), heatmap.min()]) 
        h2[heatmap==0] = np.mean([h2.max(), h2.min()]) 
        visu = plt.subplot2grid(gridsize, (0, 0), rowspan=2, colspan=4, fig=fig)
        pcm = visu.imshow(heatmap, cmap='coolwarm')
        fig.colorbar(pcm, ax=visu)
        heat1 = plt.subplot2grid(gridsize, (2, 0), rowspan=2, colspan=2, fig=fig)
        pcm = heat1.imshow(h1, cmap='coolwarm')
        heat1.set_title('scores')
        fig.colorbar(pcm, ax=heat1)
        heat2 = plt.subplot2grid(gridsize, (2, 2), rowspan=2, colspan=2, fig=fig)
        pcm = heat2.imshow(h2, cmap='coolwarm')
        fig.colorbar(pcm, ax=heat2)
        heat2.set_title('embedding norm')

        ##plt.subplot(1,2,1)
        ##plt.imshow(self.images['heatmap'], figure=fig)
        ##plt.colorbar()
        ##plt.subplot(1, 2, 2)
        ##plt.imshow(self.images['heatmap_norm'], figure=fig)
        ##plt.colorbar()
        return fig

    def _make_message(self):
        if self.result_pred is '':
            msg = "Prediction of {} : {}.".format(self.target_name, self.pred)
        else:
            msg = "Prediction of {} : {}. Ground truth: {}".format(self.target_name, self.pred, self.gt)
        return msg

        out_path = os.path.join(self.out_path, wsi_ID + '_heatmap.jpg')
        fig, _ = self.compute_heatmap(wsi_ID, embeddings, raw)
        fig.savefig(out_path)

    def _transfer_to_infomat(self, infomat):
        """transfers the weights hooked on to the infomat.

        Args:
            infomat (npy): npy array containing the index of each tile
        """
        assert self.tiles_weights is not None, "you have to compute one forward pass of the model"
        size = (infomat.shape[0], infomat.shape[1], self.num_heads)
        infomat = infomat.flatten()
        heatmap = np.zeros((len(infomat), self.num_heads))
        for o, i in enumerate(infomat):
            if i > 0:
                for nh in range(self.num_heads):
                    if len(self.tiles_weights.shape) > 1:
                        heatmap[o, :] = self.tiles_weights[int(i), :]
                    else:
                        heatmap[o, :] = self.tiles_weights[int(i)]# avec le nouveau code infomat, enlever le +1 !!! juste, erreur lors de la création infomat (14/10/20)
        heatmap = heatmap.reshape(size)
        return heatmap

    def _transfert_feature_norm_to_infomat(self, infomat):
        """
        transfers the norm of the feature vectors on to the infomat
        Mostly there for debugging AND interprete the scores
        """
        size = infomat.shape
        infomat = infomat.flatten()
        heatmap = np.zeros(len(infomat))
        for o, i in enumerate(infomat):
            if i > 0:
                heatmap[o] = np.linalg.norm(self.tiles_features, axis=1)[int(i)]
        heatmap = heatmap.reshape(size)
        heatmap[heatmap==0] = np.mean([heatmap.max(), heatmap.min()])
        return heatmap

    ## for debugging
    def plot_infomat(self, image, infomat):
        """
        Plots infomat and image superimposed
        """
        figure = plt.figure(figsize=(25, 20))
        plt.imshow(np.mean(image, axis=2), cmap='gray', interpolation='bilinear')
        plt.axis('off')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.imshow(infomat, cmap='red', interpolation='nearest', alpha=.5, extent=(xmin,xmax,ymin,ymax))   # for heatmap to overlap
        return figure

    def _get_hook(self):
        def hook_tiles(m, i, o):
            tiles_weights = o
            self.tiles_weights = tiles_weights.squeeze().detach().cpu().numpy()

        def hook_features(m, i, o):
            tiles_features = o
            self.tiles_features = tiles_features.squeeze().detach().cpu().numpy()

        def hook_reprewsi(m, i, o):
            repre = i[0].view(self.num_heads, -1)
            print(repre.shape)
            self.reprewsi = repre.squeeze().detach().cpu().numpy()

        def hooker_attentionmil(net):
            for name, layer in net.named_children():
                if list(layer.children()):
                    hooker_attentionmil(layer)
                if name == 'weight_extractor':
                    hook_layer = list(layer.children())[2]
                    hook_layer.register_forward_hook(hook_tiles)
                    print('Hook in place, captain')

        def hooker_multiheadmil(net):
            for name, layer in net.named_children():
                if list(layer.children()):
                    hooker_multiheadmil(layer)
                if name == 'attention':
                    hook_layer = list(layer.children())[0]
                    hook_layer.register_forward_hook(hook_tiles)
                    print('Hook in place, captain')
                if name == 'classifier':
                    hook_layer =  list(layer.children())[0]
                    hook_layer.register_forward_hook(hook_reprewsi)
    
        hooker_dict = {'attentionmil': hooker_attentionmil,
                'multiheadmil': hooker_multiheadmil,
                'multiheadmulticlass': hooker_multiheadmil, 
                'mhmc_layers_reg': hooker_multiheadmil,
                'mhmc_layers':hooker_multiheadmil,
                }
        hooker_dict[self.model_name](self.model.network)

    @staticmethod
    def _make_background_neutral(heatmap):
        """
        For the sake of visibility, puts the background neutral.
        """
        heatmap_neutral = copy(heatmap)
        heatmap_neutral[heatmap_neutral == 0] = np.mean([heatmap_neutral.max(), heatmap_neutral.min()])
        return heatmap_neutral

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

from model_hooker import HookerMIL


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
    def __init__(self, model, table=None, level_visu=-1):
        self.k = k
        ## Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model, self.device)
        if table is None:
            self.table = self.model.args["table_data"]

        ## Model parameters
        self.num_heads = self.model.args.num_heads
        self.model_name = self.model.args.model_name
        self.target_name = self.model.args.target_name
        self.level_visu = level_visu

        ## Instanciate hooker
        self.hooker = HookerMIL(self.model.network, self.num_heads)
        
        ## list that have to be filled with various WSI
        self.stored = {"names":[], "heads":[], "reprewsi":[]}

        self.gt = None
        self.pred = None
        self.wsi_ID = None
        self.path_emb = None
        self.path_raw = None

    def forward_pass(self, wsi_ID: str, path_emb: str) -> str:
        """forward_pass.
    
        Makes a forward pass of wsi_ID, hooks the interesting layers
        in the hooker. Gets also the info-object.
    
        :param wsi_ID: name of the wsi, without the extension
        :type wsi_ID: str
        :param path_emb: path where are stored the outputs folders of the tiling pipeline (info, mat_pca etc..).
        :type path_emb: str
        """
        self.wsi_ID = wsi_ID 
        self.path_emb = path_emb
        self.info = self._get_info(wsi_ID, path_emb)
        wsi_embedded_path = os.path.join(embeddings, 'mat_pca', wsi_ID + "_embedded.npy")
        input_wsi = self._preprocess(wsi_embedded_path)
        _, pred = self.model.predict(input_wsi)
        self.store_wsi_features()
        self.gt = self._extract_groundtruth(table=self.table, wsi_ID=wsi_ID)
        self.pred = pred
        return pred

    def make_visu(self, figtypes, wsi_ID, path_emb, path_raw, head=1):
        pred = self.forward_pass(wsi_ID, path_emb)
        figs = {}
        for ft in figtypes:
            figs[ft] = getattr(self, 'create_{}_fig'.format(ft))(path_raw, pred)
        return figs

    def store_wsi_features(self):
        self.stored["names"].append(self.wsi_ID)
        self.stored['heads'].append(self.hooker.head_average)
        self.stored['reprewsi'].append(self.hooker.reprewsi)

    def flush_wsi_features(self):
        self.stored = {"names":[], "heads":[], "reprewsi":[]}

    def create_heatmap_fig(self, path_raw)
        heatmaps = self._get_heatmaps()
        num_heads = self.num_heads
        num_cases = round((num_heads + 1)/2)+1
        gridsize = (num_cases, 2)
        fig = plt.figure(figsize=(num_cases * 10, 20))
        for nh in range(num_heads):
            r = nh // 2
            c = nh % 2
            heatarray = self._make_background_neutral(heatmaps[:,nh])
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

    def create_summary_fig(self, path_raw, head=0):
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
        hm = hm_ax.imshow(self._make_background_neutral(heatmap), cmap='coolwarm')
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
 

# %%
