from deepmil.predict import load_model
from scipy.special import softmax
from openslide import open_slide 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchvision import transforms as ttransforms
import skimage
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
try:
    from .model_hooker import HookerMIL
except:
    from tiler_wsi.tile_retriever.model_hooker import HookerMIL
import os 
import fire

def make_background_neutral(heatmap, ref_min=None, ref_max=None):
    """
    For the sake of visibility, puts the background neutral.
     = puts the background at the center of the heatmap distri.
    """
    heatmap_neutral = copy(heatmap)
    ref_min = heatmap_neutral.min() if ref_min is None else ref_min
    ref_max = heatmap_neutral.max() if ref_max is None else ref_max
    heatmap_neutral[heatmap_neutral == 0] = np.mean([ref_min, ref_max])
    return heatmap_neutral

def add_titlebox(ax, text):
    ax.text(.05, .05, text,
        horizontalalignment='left',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        fontsize=20)
    return ax

class BaseVisualizer(ABC):
    def __init__(self, model):
        ## Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model, self.device)
        self.classifier = self.model.network.mil.classifier
        self.classifier.eval()

        self.target_correspondance = self.model.target_correspondance

        # Par défaut on considère le dataset d'entrainement 
        # = données contenues dans args.
        self.table = self._load_table(self.model.table_data)
        self.path_emb = self.model.args.wsi
        self.path_raw = self.model.args.raw_path
        self.num_class = self.model.args.num_class
        self.num_heads = self.model.args.num_heads
        self.target_name = self.model.args.target_name
        self.hooker = HookerMIL(self.model.network, self.model.args.num_heads)

    def _load_table(self, table):
        warning_msg = "         Carefull :          \n"
        warning_msg +="you are loading a table_data from path \n"
        warning_msg +="the test attribution of data might be different \n"
        warning_msg +="used during training.  \n"
        warning_msg +="Performances might be overestimated."
        if type(table) is str:
            print(warning_msg)
            table = pd.read_csv(table)
        return table

    def _get_info(self, wsi_ID, path_emb):
        wsi_info_path = os.path.join(path_emb, 'info')
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy')
        infomat = np.load(infomat)
        infomat = infomat.T 
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
            infodict = self.add_name_to_dict(infodict, wsi_ID) 
        return {'infomat':infomat, 'paradict': infodict}
    
    def add_name_to_dict(self, dico, name):
        """
        because the name of the slide is not included in the dico.
        """
        for d in dico:
            dico[d]['name'] = name
        return dico

    def _get_image(self, path_raw, info):
        """_get_image.
        extract the indice-ieme tile of the wsi stored at path_raw.
        returns PIL image.

        :param path_raw: path to te wsi (all extensions accepted)
        :param indice: number of the tile in the flattened WSI
        """
        param = info
        path_wsi = glob(os.path.join(self.path_raw, param['name'] + '.*'))
        assert path_wsi, "no wsi with name {}".format(param['name'])
        assert len(path_wsi)<2, "several wsi with name {}".format(param['name'])
        slide = open_slide(path_wsi[0])
        image = slide.read_region(location=(param['x'], param['y']),
                                level=param['level'], 
                                size=(param['xsize'], param['ysize']))
        return image

    def _preprocess(self, input_path, expand_bs=False):
        """preprocess the input to feed the model
    
        Args:
            input_path (str): str to the input path 
        """
        depth = self.model.args.feature_depth
        inp = np.load(input_path)[:,:depth]
        inp = torch.Tensor(inp)
        inp = inp.unsqueeze(0) if expand_bs else inp
        inp = inp.to(self.device)
        return inp

    def set_data_path(self, path_raw=None, path_emb=None, table=None):
        self.path_raw = path_raw if path_raw is not None else self.path_raw
        self.path_emb = path_emb if path_raw is not None else self.path_emb
        self.table = table if table is not None else self.table

    @abstractmethod
    def forward(self, wsi_ID):
        pass

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

# TODO reimplement using the BaseVisualizer class
class VisualizerMIL(BaseVisualizer):
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
    def __init__(self, model, level_visu=-1):
        super(VisualizerMIL, self).__init__(model)
        ## Model parameters
        self.level_visu = level_visu
        
        ## list that have to be filled with various WSI
        self.stored = {"names":[], "heads":[], "reprewsi":[], 'label':[], 
                'pred':[], 'tiles_weights':[], 'logits_tiles':[]}
        self.gt = None
        self.pred = None
        self.wsi_ID = None

    def forward(self, wsi_ID):
        """forward_pass.
    
        Makes a forward pass of wsi_ID, hooks the interesting layers
        in the hooker. Gets also the info-object.
    
        :param wsi_ID: name of the wsi, without the extension
        :type wsi_ID: str
        :param path_emb: path where are stored the outputs folders of the tiling pipeline (info, mat_pca etc..).
        :type path_emb: str
        """
        self.wsi_ID = wsi_ID 
        self.info = self._get_info(wsi_ID, self.path_emb)
        wsi_embedded_path = os.path.join(self.path_emb, 'mat_pca', wsi_ID + "_embedded.npy")
        input_wsi = self._preprocess(wsi_embedded_path, expand_bs=True)
        _, pred = self.model.predict(input_wsi)
        self.gt = self._extract_groundtruth(table=self.table, wsi_ID=wsi_ID)
        self.pred = pred
        self.store_wsi_features()
        return pred

    def forward_classifier(self, wsi_ID):
        wsi_embedded_path = os.path.join(self.path_emb, 'mat_pca', wsi_ID + "_embedded.npy")
        input_wsi = self._preprocess(wsi_embedded_path, expand_bs=False)
        self.classifier(input_wsi)
 

    def create_masks(self, N=500, head=0):
        """create_masks.

        Creates binary masks in order to sample at higher resolution.
        must follow a forward pass.

        :param N: number of tiles at the higher resolution
        :param head: attention head to use to class the tiles.
        """
        ## Making the heatmaps
        heatmap = self._get_heatmap()
        heatmap = heatmap[:,:,head]
        indices = np.argsort(heatmap.flatten())
        mask = np.zeros(len(heatmap.flatten()))
        mask[indices[-N:]] = 1
        mask = mask.reshape(heatmap.shape)
        image = self._get_down_image(os.path.join(self.path_raw,self.wsi_ID))
        mask_up = skimage.transform.resize(mask, image[:,:,0].shape)
        return mask_up
        
    def make_visu(self, figtypes, wsi_ID):
        if type(figtypes) == str:
            figtypes = figtypes.split(',')
        if type(wsi_ID) is int:
            x = glob(os.path.join(self.path_emb, 'mat_pca', '*_embedded.npy'))[wsi_ID]
            wsi_ID = os.path.basename(x).split('_embedded')[0]
        pred = self.forward(wsi_ID)
        figs = {}
        # Faut la mettre a la fin sinon reprewsi mauvaise.
        if 'heatmap_target' in figtypes:
            figtypes.append(figtypes.pop(figtypes.index('heatmap_target')))
        for ft in figtypes:
            figs[ft] = getattr(self, 'create_{}_fig'.format(ft))()
        return self#figs

    def store_wsi_features(self):
        self.stored["names"].append(self.wsi_ID)
        self.stored['heads'].append(self.hooker.head_average)
        self.stored['reprewsi'].append(self.hooker.reprewsi)
        self.stored['label'].append(self.gt)
        self.stored['pred'].append(self.pred)
        self.stored['tiles_weights'].append(self.hooker.tiles_weights)

    def store_logits_per_tiles(self):
        self.stored['logits_tiles'].append(self.hooker.scores)

    def flush_wsi_features(self):
        self.stored = {"names":[], "heads":[], "reprewsi":[], "label":[], 'pred':[], 'tiles_weights':[], 'logits_tiles':[]}
        
    def create_heatmap_target_fig(self):
        """
        adapted heatmap giving the scores  * logits of each tiles for each of
        the target classes.
        Ne peut etre utilisée que avec num_heads=1.
        """
        # Second  forward pass
        assert self.num_heads == 1, 'You can create the target_heatmap only with a one head attentionMIL.'
        self.forward_classifier(self.wsi_ID)
        self.store_logits_per_tiles()
        path_raw = self.path_raw
        fig, axes = plt.subplots(self.num_class, figsize=(10,10))
        target_scores = np.multiply(self.hooker.scores, self.hooker.tiles_weights)
        for o,ax in enumerate(axes):
            target_score = target_scores[:,o]
            #target_score = self.hooker.tiles_weights
            heatmap = make_background_neutral(self._decalcomanie(self.info['infomat'], target_score),
                    ref_min=target_scores.min(), ref_max=target_scores.max())
            hm = ax.imshow(heatmap, cmap='coolwarm', vmin=target_scores.min(), vmax=target_scores.max())
            ax.set_title('heatmap for {}'.format(self.target_correspondance[o]))
            fig.colorbar(hm, ax=ax)
            ax.set_axis_off()
        add_titlebox(ax, self._make_message(self.pred))
        fig.tight_layout()
        return fig

    def create_heatmap_fig(self, path_raw=None):
        if path_raw is None:
            path_raw = self.path_raw
        heatmaps = self._get_heatmap()
        num_heads = self.num_heads
        num_cases = round((num_heads + 1)/2)+1
        gridsize = (num_cases, 2)
        fig, axes = plt.subplots(self.num_heads + 1, figsize=(10, 10))
        for o,ax in enumerate(axes):
            if o > (self.num_heads-1):
                ax.imshow(self._get_down_image(os.path.join(self.path_raw, self.wsi_ID)))
                ax.set_axis_off()
            else:
                heatarray = make_background_neutral(heatmaps[:,:,o])
                hm = ax.imshow(heatarray, cmap='coolwarm', vmin=heatarray.min())
                ax.set_title('Scores')
                ax.set_axis_off()
        add_titlebox(ax, self._make_message(self.pred))
        fig.tight_layout()
        return fig

    def create_summary_fig(self, path_raw=None, head=0):
        if path_raw is None:
            path_raw = self.path_raw
        color_tile = {0:'red', 1:'blue'}
        legend_elements = [patches.Patch(facecolor=color_tile[0], edgecolor='black', label='Highest scores'), 
                          patches.Patch(facecolor=color_tile[1], edgecolor='black', label='Lowest scores')]

        ## Making the heatmaps
        heatmap = self._get_heatmap()
        heatmap = heatmap[:,:,head]

        gridsize = (6, 4)
        fig = plt.figure(figsize=(30, 20))
        visu = plt.subplot2grid(gridsize, (2, 0), rowspan=2, colspan=2, fig=fig)
        visu.imshow(self._get_down_image(os.path.join(self.path_raw, self.wsi_ID)))
        visu.set_axis_off()
        hm_ax = plt.subplot2grid(gridsize, (2, 2), rowspan=2, colspan=2, fig=fig)
        hm = hm_ax.imshow(make_background_neutral(heatmap), cmap='coolwarm')
        hm_ax.set_title('Scores')
        fig.colorbar(hm, ax=hm_ax)
        hm_ax.set_axis_off()

        # Best/worst tiles
        topk = {}
        topk['bestk'], topk['lowk'] = self._get_topk_tiles(heatmap)
        tiles = []
        for l,L in enumerate(['bestk', 'lowk']):
            for c in range(4):
                ax = plt.subplot2grid(gridsize, (l,c), fig=fig)
                ind = topk[L][c]
                im = self._get_image(path_raw, ind)
                ax.imshow(im)
                ax = set_axes_color(ax, color=color_tile[l]) 
                tiles.append(ax)
                visu = self._plot_loc_tile(visu, color=color_tile[l], para=self.info['paradict'][c])
        visu.legend(handles=legend_elements, loc='upper right', fontsize=12, handlelength=2)
        add_titlebox(visu, self._make_message(self.pred))
        fig.tight_layout()
        return fig

    def get_best_tile(self, path_raw=None, head=0):
        if path_raw is None:
            path_raw = self.path_raw
        heatmap = self._get_heatmap()[:,:,head]
        top, _ = self._get_topk_tiles(heatmap, k=1)
        top_tile = self._get_image(path_raw, top)
        return top_tile

    def _get_topk_tiles(self, heatmap, k=4):
        infomat = self.info['infomat'].flatten()
        mask = infomat > -1
        heatmap = heatmap.flatten()[mask]
        indices = np.argsort(heatmap)
        topk = indices[-k:]
        lowk = indices[:k]
        topk_i = infomat[mask][topk]
        lowk_i = infomat[mask][lowk]
        return topk_i, lowk_i

    def _get_down_image(self, wsi):
       """get the downsampled image (numpy format) at the desired downsampling factor.
       """
       self.wsi = usi.open_image(wsi+'.ndpi')
       if self.level_visu < 0:
           self.level_visu = self.wsi.level_count + self.level_visu
       image = usi.get_whole_image(self.wsi, level=self.level_visu, numpy=True)
       return image

    def _decalcomanie(self, infomat, tile_scores):
        """_decalcomanie.
        trasnfers the scores of tile_scores to the infomat. Infomat is 
        a downsampled map of the WSI. Its elements give the index of each tile.

        :param infomat: mxn array, (infomat > 0).sum()==len(tile_scores)
        :param tile_scores: scores to décalque on the infomat !
        """
        size = infomat.shape
        infomat = infomat.flatten()
        heatmap = np.zeros(len(infomat))
        for o, i in enumerate(infomat):
            if i > 0:
                heatmap[o] = tile_scores[int(i)]
        heatmap = heatmap.reshape(size)
        return heatmap
       
    def _get_heatmap(self):
        assert self.wsi_ID, "You need to forward pass a WSI before"
        tiles_weights =  self.hooker.tiles_weights
        infomat = self.info['infomat']
        heatmaps = []
        for nh in range(self.num_heads):
            heatmaps.append(self._decalcomanie(infomat, tiles_weights[:,nh]))
        heatmaps = np.concatenate(heatmaps, axis=-1).reshape((infomat.shape[0], infomat.shape[1], self.num_heads))
        return heatmaps

    def _get_info(self, wsi_ID, path_emb):
        wsi_info_path = os.path.join(path_emb, 'info')
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy')
        infomat = np.load(infomat)
        infomat = infomat.T 
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
        return {'infomat':infomat, 'paradict': infodict , 'paralist': [self._infodict_to_list(infodict[x]) for x in infodict]}

    def _get_image(self, raw_path, indice):
        # TODO Change the loader to get rid of usi.
        para = self.info['paralist'][int(indice.item())]
        image = usi.get_image(slide=os.path.join(self.path_raw, self.wsi_ID+'.ndpi'), 
                para=para, numpy=False)
        return image

    def _plot_loc_tile(self, ax, color, para):
         args_patch = {'color':color, 'fill': False, 'lw': 5}
         top_left_x, top_left_y = usi.get_x_y_from_0(self.wsi, (para['x'], para['y']), self.level_visu)
         width, height = usi.get_size(self.wsi, (para['xsize'], para['ysize']), para['level'], self.level_visu)
         plot_seed = (top_left_x, top_left_y)
         patch = patches.Rectangle(plot_seed, width, height, **args_patch)
         ax.add_patch(patch)
         return ax
  
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

    def _make_message(self, pred):
        if pred is '':
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
        else:
            return "not in table"
        return gt
 
class TileSeeker(BaseVisualizer): 
    """
    Seeks for the best tile scoring tile according to one class.
    """
    def __init__(self, model, n_best, min_prob=False):
        super(TileSeeker, self).__init__(model)
        self.attention = self.model.network.mil.attention
        self.attention.eval()
        self.n_best = n_best
        self.min_prob = min_prob

        ## Model parameters
        assert self.model.args.num_heads == 1, 'you can\'t extract a best tile when using the multiheaded attention'
        self.model_name = self.model.args.model_name
        self.target_name = self.model.args.target_name

        ## list that have to be filled with various WSI
        self.store_info = None
        self.store_score = None
        self.store_images = None
        self._reset_storage()

    def forward(self, wsi_ID):
        """forward.
        Execute a forward pass through the MLP classifier. 
        Stores the n-best tiles for each class.
        :param wsi_ID: wsi_ID as appearing in the table_data.
        """
        if type(wsi_ID) is int:
            x = glob(os.path.join(self.path_emb, 'mat_pca', '*_embedded.npy'))[wsi_ID]
            wsi_ID = os.path.basename(x).split('_embedded')[0]
        else:
            x = os.path.join(self.path_emb,'mat_pca', wsi_ID+'_embedded.npy')
        info = self._get_info(wsi_ID, path_emb=self.path_emb)

        #process each images
        x = self._preprocess(x)
        out = self.classifier(x) # (bs, n_class)
        out = out.detach().cpu()
        logits = self.hooker.scores

        self.attention(x.unsqueeze(0))
        tw = self.hooker.tiles_weights.squeeze()
        _, ind = torch.sort(torch.Tensor(tw))
        size_select = min(500, len(ind))
        selection = set(ind[-size_select:].cpu().numpy())
        
        # Find attention scores to filter out of distribution tiles
#        out = self._postprocess(out)
        self.store_best(logits, info, selection, min_prob=self.min_prob)
        return self

    def forward_all(self, test=True):
        table = self.table
        test = int(self.model.args.test_fold)
        wsi_ID = table[table['test'] == test]['ID'].values
        for o in wsi_ID:
            try:
                print(o)
                self.forward(o)
            except:
                continue
        return self

    def _postprocess(self, out):
#        sc = StandardScaler()
#        out = sc.fit_transform(out)
        out = np.array(torch.nn.functional.softmax(torch.Tensor(out/100), dim=-1))
        return out

    def store_best(self, out, info, selection, min_prob=False):
        """store_best.
        decides if we have to store the tile, according the final activation value.

        :param out: out of a forward parss
        :param info: info dictionnary of the WSI
        :param min_prob: bool:
        maximiser proba -> prendre les derniers éléments de indices_best (plus grand au plus petit)
        minimiser proba -> prendre les premiers éléments de indice_best
        """
        sgn = -1 if min_prob else 1
        # for each tile
        for s in range(out.shape[0]): 
            if s not in selection:
                continue
            # for each class
            for o in range(out.shape[1]):
                # If the score for class o at tile s is bigger than the smallest 
                # stored value: put in storage
                if (len(self.store_score[o]) < self.n_best) or (sgn * out[s,o] >= sgn * self.store_score[o][0]):
                    self.store_info[o].append(info['paradict'][s])
                    self.store_score[o].append(out[s,o])

        #trie et enleve les plus bas
        # Permute storage pour que les probamax (=1) ne soient pas issues toutes
        # de la meme slide
        self._permute_storage() 
        for o in range(out.shape[1]):
            indices_best = np.argsort(self.store_score[o])[::sgn][-self.n_best:]
            self.store_score[o] = list(np.array(self.store_score[o])[indices_best])
            self.store_info[o] = list(np.array(self.store_info[o])[indices_best])
                
    def _permute_storage(self):
        for o in range(len(self.store_score)):
            permutation = np.random.permutation(np.arange(len(self.store_score[o])))
            self.store_score[o] = list(np.array(self.store_score[o])[permutation])
            self.store_info[o] = list(np.array(self.store_info[o])[permutation])

    def _reset_storage(self):
        """_reset_storage.
        Reset the storage dict.
        store_score and store info are dict with keys the classes (ordinals)
        containing empty lists. When filled, they are supposed to n_best scores 
        and infodicts values.
        only store images as the name of the targets as keys. 
        Advice : fill store image at the end only.
        """
        self.store_score = dict()
        self.store_info = dict()
        self.store_image = dict()
        for o,i in enumerate(self.target_correspondance):
            self.store_info[o] = []
            self.store_score[o] = []
            self.store_image[i] = []

    def extract_images(self):
        for o,i in enumerate(self.target_correspondance):
            assert self.store_score[o], "no tile found"
            print(len(self.store_score[o]))
            self.store_image[i] = [self._get_image(self.path_raw, x).convert('RGB') for x in self.store_info[o]]
            # Print origine des tuiles
            print([x['name'] for x in self.store_info[o]])
            print(self.store_score[o])
        return self

if __name__ == '__main__':
#    fire.Fire(TileSeeker)
    fire.Fire(VisualizerMIL)
    






