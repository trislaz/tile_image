import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import preprocessing
import seaborn as sns
import numpy as np; np.random.seed(42)
from argparse import ArgumentParser
import os
import pandas as pd
import os; from glob import glob; import numpy as np; import pickle; import matplotlib.pyplot as plt; import seaborn as sns; import pandas as pd; import umap

def main(mean_tile):
    le = preprocessing.LabelEncoder()
    projection = np.load('projection_reprewsi.npy')
    names = np.load('names.npy')
    labels = np.load('labels.npy') #le.fit_transform(np.load('labels.npy'))
    preds = np.load('preds.npy')
    ## Arrange the jpg_name_np as for the scatter plot.
    cmap = plt.cm.RdYlGn
    
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line = sns.scatterplot(x=projection[:,0], y=projection[:,1], hue=labels, s=40, ax=ax)
    line = line.collections[0]
#    line = ax.scatter(x=projection[:,0], y=projection[:,1], c=labels, s=40)
    if mean_tile:
        directory = 'representants'
    else:
        directory = 'best_tiles'
    image_path = np.asarray(['{}/{}_{}.png'.format(directory, preds[o], x) for o,x in enumerate(names)])
    
    # create the annotations box
    image = plt.imread(image_path[0])
    im = OffsetImage(image, zoom=0.7)
    xybox=(120., 120.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            print(line.contains(event)[1]["ind"])
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(projection[ind,0], projection[ind, 1])
            # set the image corresponding to that point
            im.set_data(plt.imread(image_path[ind]))
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()
    
    # add callback for mouse moves
 #   fig.canvas.mpl_connect('button_press_event', hover)  
    fig.canvas.mpl_connect('motion_notify_event', hover)  
    fig = plt.gcf()
    fig.set_size_inches(10.5, 9.5)
    plt.show() 

if __name__ == '__main__':

    parser=ArgumentParser()
    parser.add_argument('--mean_tile', action='store_true', help='images are mean tiles')
    args = parser.parse_args()
    main(mean_tile=args.mean_tile)
