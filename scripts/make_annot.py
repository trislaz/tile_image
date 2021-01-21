import numpy as np
import useful_wsi as usi
from skimage.draw import polygon

def parse_wkt(wkt):
    XY = []
    str_coord = wkt.split('(')[2].split(')')[0]
    str_coord = str_coord.split(',')
    for c in str_coord:
        if c:
            x, y =[x for x in c.split(' ') if x]
            XY.append((np.float(x), np.float(y)))
    return XY

def make_annot(wsi, wkt, level_mask):
    slide = usi.open_image(wsi) 
    size = slide.level_dimensions[-level_mask]
    mask = np.zeros(size)
    XY = parse_wkt(wkt)
    XY = [usi.get_x_y_from_0(slide, x, -4) for x in XY]
    X = [x[0] for x in XY]
    Y = [x[1] for x in XY]
    rr, cc = polygon(X, Y)
    mask[rr, cc] = 1
    return mask

    

