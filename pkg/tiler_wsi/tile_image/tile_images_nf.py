#%%
import useful_wsi as usi
from glob import glob
from argparse import ArgumentParser
from torchvision.models import resnet50, resnet18, resnext50_32x4d, densenet121
from torchvision import transforms
from torch.nn import Identity
import torch
import pandas as pd
import pickle
import os
import numpy as np
from PIL import Image
from xml.dom import minidom
from tqdm import tqdm
from skimage.draw import polygon
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage._shared.utils import warn

## Encore cette histoire de labels à gérer.

def threshold_otsu(image, mask, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    raveled_image = image[mask > 0].ravel()
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    if raveled_image.min() == raveled_image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(raveled_image.min()))

    hist, bin_centers = histogram(raveled_image, nbins)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def get_polygon(image, path_xml, label):
    
    doc = minidom.parse(path_xml).childNodes[0]
    nrows = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('nrows')[0].firstChild.data
    ncols = doc.getElementsByTagName('imagesize')[0].getElementsByTagName('ncols')[0].firstChild.data
    size_image = (image.shape[0], image.shape[1])
    mask = np.zeros(size_image)
    obj = doc.getElementsByTagName('object')
    polygons = []
    for o in obj:
        if True:#o.getElementsByTagName('name')[0].firstChild.data == label:
            polygons += o.getElementsByTagName('polygon')
            print(polygons)
    if not polygons:
        raise ValueError('There is no annotation with label {}'.format(label))

    for poly in polygons:
        rows = []
        cols = []
        for point in poly.getElementsByTagName('pt'):
            x = int(point.getElementsByTagName('x')[0].firstChild.data)
            y = int(point.getElementsByTagName('y')[0].firstChild.data)
            rows.append(y)
            cols.append(x)
        rr, cc = polygon(rows, cols)
        mask[rr, cc] = 1
    return mask

def make_label_with_otsu(xml_file, rgb_img):
    mask = get_polygon(rgb_img, xml_file,label='t')
    grey_rgb = rgb2gray(rgb_img)
    thresh = threshold_otsu(grey_rgb, mask, nbins=256)
    binary = (grey_rgb < thresh).astype(float)
    merge_mask = mask + binary
    merge_mask[merge_mask != 2] = 0
    merge_mask[merge_mask > 0] = 1
    merge_mask = dilation(merge_mask)
    return merge_mask

class ImageTiler:
    def __init__(self, args, make_info=True, nf=True):
        self.level = args.level # Level to which sample patch.
        self.use_nf = nf # Useful for managing the outputs
        self.sampling_method = 'grid'# grid
        self.n_samples = 15 # pour les slides 0.
        self.device = args.device
        self.size = (args.size, args.size)
        self.path_wsi = args.path_wsi 
        self.path_outputs = args.path_outputs
        self.from_0 = args.from_0
        self.auto_mask = args.auto_mask
        self.path_mask = args.path_mask
        self.model_path = args.model_path 
        self.infomat = None
        self.prediction_map = None
        self.mask = None
        self.tiler = args.tiler
        self.name_wsi, self.ext_wsi = os.path.splitext(os.path.basename(self.path_wsi))
        self.path_info = '.' if nf else os.path.join(self.path_outputs, 'info') 
        self.path_visu = '.' if nf else os.path.join(self.path_outputs, 'visu')
        self.path_tiles = '.' if nf else os.path.join(self.path_outputs, self.name_wsi)
        self.slide = usi.open_image(self.path_wsi)
        self.make_info = make_info
        if args.mask_level < 0:
            self.mask_level = self.slide.level_count + args.mask_level
        else:
            self.mask_level = args.mask_level    
        self.rgb_img = usi.get_whole_image(self.slide, level=self.mask_level, numpy=True)
        self.mask_tolerance = args.mask_tolerance

    def _get_mask_function(self):
        """
        the patch sampling functions need as argument a function that takes a WSI a returns its 
        binary mask, used to tile it. here it is.
        """
        if self.path_mask == "no":
            if self.mask is not None:
                mask_function = lambda x: self.mask
            else:
                mask_function = lambda x: np.ones(self.rgb_img.shape[0:-1])
        else:
            if self.auto_mask:
                path_mask = os.path.join(self.path_mask, self.name_wsi + ".npy")
                assert os.path.exists(path_mask), "The mask should carry the same name as its WSI"
                mask_function = lambda x: np.load(path_mask)
        
            else:
                path_mask = os.path.join(self.path_mask, self.name_wsi + ".xml")
                assert os.path.exists(path_mask), "The mask should carry the same name as its WSI"
                mask_function = lambda x: get_polygon(image=self.rgb_img, path_xml=path_mask, label='t')
        return mask_function

    def tile_image(self):
        self.mask_function = self._get_mask_function()
        tiler = getattr(self, self.tiler + '_tiler')
        if self.from_0:
            level = 0
            patch_size = ((self.level * 2) * self.size[0], (self.level * 2)*self.size[1])
        else:
            level = self.level
            patch_size = self.size
        param_tiles = usi.patch_sampling(slide=self.slide, mask_level=self.mask_level, mask_function=self.mask_function, 
            sampling_method=self.sampling_method, n_samples=self.n_samples, analyse_level=level, patch_size=patch_size, mask_tolerance = self.mask_tolerance)
        if self.make_info:
            self._make_infodocs(param_tiles)
            self._make_visualisations(param_tiles)
        tiler(param_tiles)
   
    def _make_visualisations(self, param_tiles):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                     'title': "n_tiles={}".format(len(param_tiles))}
        usi.visualise_cut(self.slide, param_tiles, res_to_view=self.mask_level, plot_args=PLOT_ARGS)
        plt.savefig("{}_visu.png".format(os.path.join(self.path_visu,self.name_wsi)))

    def _get_infomat(self):
        """Returns a zeros matrix, such that each entry correspond to a tile in the WSI.
        I will stock there the ID of each tile. 

        Returns
        -------
        tuple
            (mat -array- , size_patch_0 -int, size of a patch in level 0- )
        """
        size_patch_0 = usi.get_size(self.slide, size_from=self.size, level_from=self.level, level_to=0)[0]
        dim_info_mat = (self.slide.level_dimensions[0][0] // size_patch_0, self.slide.level_dimensions[0][1] // size_patch_0)
        info_mat = np.zeros(dim_info_mat)
        return info_mat, size_patch_0 

    def _make_infodocs(self, param_tiles):
        infodict = {}
        infos=[]
        infomat , patch_size_0 = self._get_infomat() 
        if self.tiler == 'classifier':
            self.infomat_classif = np.zeros(infomat.shape)
        for o, para in enumerate(param_tiles):
            infos.append({'ID': o, 'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]})
            infodict[o] = {'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]} 
            infomat[para[0]//(patch_size_0+1), para[1]//(patch_size_0+1)] = o+1 #+1 car 3 lignes plus loin je sauve infomat-1 (background à -1) 
        df = pd.DataFrame(infos)
        df.to_csv(os.path.join(self.path_info, self.name_wsi + '_infos.csv'), index=False)
        np.save(os.path.join(self.path_info, self.name_wsi + '_infomat.npy'), infomat-1)
        with open(os.path.join(self.path_info,  self.name_wsi + '_infodict.pickle'), 'wb') as f:
            pickle.dump(infodict, f)
        self.infomat = infomat -1
        self.prediction_map = np.zeros(infomat.shape) - 1
   
    def get_new_path(self, name, subfolder):
       if self.use_nf:
           return os.path.join('.', name)
       else:
           return os.path.join(self.path_outputs, subfolder, name)

    def simple_tiler(self, param_tiles):
        param_tiles = np.array(param_tiles)
        param_tiles = param_tiles[np.random.choice(range(len(param_tiles)), min(1000,len(param_tiles)), replace=False)]
        for o, para in enumerate(param_tiles):
            patch = usi.get_image(slide=self.path_wsi, para=para, numpy=False)
            if self.from_0:
                patch = patch.resize(self.size)
            patch = patch.convert('RGB')
            name_wsi = os.path.splitext(os.path.basename(self.path_wsi))[0]
            new_name =  "{}_{}.png".format(name_wsi, o)
            new_path = self.get_new_path(new_name, 'simple')
            patch.save(new_path)
            del patch

    def imagenet_tiler(self, param_tiles):
        model = resnet50(pretrained=True)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tiles = []
        #param_tiles = np.array(param_tiles)[np.random.choice(range(len(param_tiles)), min(4000,len(param_tiles)), replace=False)]
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            if self.from_0:
                image = image.resize(self.size)
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
            tiles.append(t.cpu().numpy())
        mat = np.vstack(tiles)
        np.save(os.path.join(self.path_outputs, '{}_embedded.npy'.format(self.name_wsi)), mat)

    def _transfer_to_prediction_map(self, pred, o):
        """
        pred = prediction of the classifier
        o = id of the tile
        used with the classifier tiler.
        """
        assert self.infomat is not None, "Pas d'infomat !"
        index = np.where((self.infomat == o).flatten())[0]
        r,c = np.unravel_index(index, shape=self.infomat.shape)
        self.prediction_map[r,c] = pred

    def imagenet_classifier_tiler(self, param_tiles):
        """
        encodes the tiles with imagenet AND heatmap (predmap) with the binary classifier.
        """
        model = resnet18()
        model.fc = torch.nn.Linear(512, 2)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k.replace('module.', '')] = state_dict[k]
                del state_dict[k]

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        preprocess = self._get_transforms(imagenet=True) 
        tiles = []
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            image = preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                proba = torch.nn.functional.softmax(model(image).squeeze(), dim=0)
                pred = proba[0].item()
            self._transfer_to_prediction_map(pred, o)
        np.save(os.path.join(self.path_outputs, '{}_predmap.npy'.format(self.name_wsi)), self.prediction_map)
        self.imagenet_tiler(param_tiles)

    def classifier_tiler(self, param_tiles):
        model = resnet18()
        model.fc = torch.nn.Linear(512, 2)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k.replace('module.', '')] = state_dict[k]
                del state_dict[k]

        model.load_state_dict(checkpoint['state_dict'])
        self._get_hook(model)
        model = model.to(self.device)
        model.eval()
        preprocess = self._get_transforms(imagenet=True) 
        tiles = []
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            image = preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                proba = torch.nn.functional.softmax(model(image).squeeze(), dim=0)
                pred = proba[0].item()
            tiles.append(self.tile_features)
            self._transfer_to_prediction_map(pred, o)
        mat = np.vstack(tiles)
        np.save(os.path.join(self.path_outputs, '{}_embedded.npy'.format(self.name_wsi)), mat)
        np.save(os.path.join(self.path_outputs, '{}_predmap.npy'.format(self.name_wsi)), self.prediction_map)

    def _get_transforms(self, imagenet=True):
        if not imagenet:
            if self.size[0] == 256:
                trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.753, 0.584,  0.706], [0.128, 0.157, 0.122])])  
            elif self.size[0] == 224:
                print('ok_norm')
                #trans = transforms.Compose([transforms.ToTensor(),
                #    transforms.Normalize([0.747, 0.515,  0.70], [0.145, 0.209, 0.154])])
                ## trans moco equi
                trans = transforms.Compose([transforms.ToTensor(), 
                    transforms.Normalize([0.703, 0.475,  0.664], [0.162, 0.183, 0.140])])
                # trans moco curie 256

            elif self.size[0] == 512:
                trans = transforms.Compose([transforms.ToTensor(), 
                    transforms.Normalize([0.763, 0.602, 0.718], [0.134, 0.174, 0.131])])
        else: 
            trans = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return trans

    def moco_tiler(self, param_tiles):
        model = resnet18()
        #model = resnext50_32x4d()
        #model = Equinet.Wide_ResNet()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = self._get_transforms(imagenet=False)
        tiles = []
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            if self.from_0:
                image = image.resize(self.size)
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
            tiles.append(t.cpu().numpy())
        mat = np.vstack(tiles)
        np.save(os.path.join(self.path_outputs, '{}_embedded.npy'.format(self.name_wsi)), mat)
        to_write = np.array(param_tiles)[np.random.choice(range(len(param_tiles)), min(5,len(param_tiles)), replace=False)]
        self.simple_tiler(to_write)

    def ciga_tiler(self, param_tiles):
        def load_model_weights(model, weights):
            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)
            return model
        model = resnet18()
        state = torch.load(self.model_path, map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        tiles = []
        #param_tiles = np.array(param_tiles)[np.random.choice(range(len(param_tiles)), min(4000,len(param_tiles)), replace=False)]
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            if self.from_0:
                image = image.resize(self.size)
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
            tiles.append(t.cpu().numpy())
        mat = np.vstack(tiles)
        np.save(os.path.join(self.path_outputs, '{}_embedded.npy'.format(self.name_wsi)), mat)



    def imagenet_v2_tiler(self, param_tiles):
        """Same as imagenet tiler, but save all the tiles in a different file

        Args:
            param_tiles (list): list of the parameters of each tiles (x, y, x0, y0, size)
        """
        model = resnet18(pretrained=True)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tiles = []
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
                t = t.cpu().numpy()
                np.save(os.path.join(self.path_outputs, 'tile_{}.npy'.format(o)), t)

    def simclr_tiler(self, param_tiles):
        raise NotImplementedError

    def _get_hook(self, model):
        def hook_tiles(m, i, o):
            tile_features = o
            self.tile_features = tile_features.squeeze().detach().cpu().numpy()

        def hooker_resnet(net):
            for name, layer in net.named_children():
                if name == 'avgpool':
                    layer.register_forward_hook(hook_tiles)
                    print('Hook in place, captain')
        hooker_resnet(model)






