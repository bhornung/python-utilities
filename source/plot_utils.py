"""
I do not like seaborn.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

from stat_utils import calculate_histo_2d

def plot_lines(xs, ys, labels, ax, kind = 'plot', cmap = plt.get_cmap('Blues'), **kwargs):
    """
    Plots a series of lines on an axis.
    Parameters:
        xs ([array-like]) : abscissa values
        yx ([array-like]) : ordinate values
        ax (axis) : axis to plot on
        kind ({'plot', 'scatter', 'step'}) : kind of plot. Allowed kinds 'plot', 'scatter', 'step'. Default: plot
    """
    
    for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
        color = cmap( 0.0 + (idx + 1) / (1 * len(xs)) )
        if kind == 'plot':
            ax.plot(x, y, label = label, c = color)
            
        elif kind == 'scatter':
            ax.scatter(x, y, label = label, c = color)
        
        elif kind == 'step':
            ax.plot(x, y, label = label, c = color, drawstyle = 'steps' )
            
        else:
            raise ValueError("'kind' must be 'plot', 'scatter', 'step'. Got: {0}".format(kind))
            
    ax.legend(ncol = 2)
    ax.grid()
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
        
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
        
    if 'xticks' in kwargs:
        ax.set_xticks(kwargs['xticks'])
        
    ax.set_xlabel(kwargs.get('xlabel', ""))
    ax.set_ylabel(kwargs.get('ylabel', ""))
        
    return 


def plot_raw_histo(x, y, ycum, ax, 
                   x1_label = None, y1_label = None, y2_label = None):
    """
    Plots a no frills 1D histogram with cumulative distribution function.
    One off helper function.
    Parameters:
        x (array-like) : abscissa values
        y (array-like) : counts
        ycum (array-like) : cumulative distribution function
        ax (axis) : axis to plot on
        x1_label (str) : label of the abscissa
        y1_label (str) : label of the ordinate
        y2_label (str) : label of the ordinate of the cumulative counts
    """
    
    ax.bar(x, y, width = 0.05, alpha = 0.4, color = 'green')
    
    ax.set_yscale('log'); ax.grid(True)
    ax.set_xticklabels([str(int(10**xt)) for xt in ax.get_xticks()])
    ax.set_xlabel(x1_label); ax.set_ylabel(y1_label, color = 'green')
    
    ax2 = ax.twinx()
    ax2.plot(x, ycum, linewidth = 2, color = 'teal')
    
    ax2.set_ylabel(y2_label, color = 'teal')
    ax2.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color = 'teal')
    
    
def add_plots_to_axis(ax, xycs, **kwargs):
    """
    Plots simple lines to an axis.
    Parameters:
        ax (axis) axis. It usually belongs to a prepared figure.
        xycs ([array-like of float, array-like of float, array-like of str]) : the abscissa, ordinate and color values.
        kwargs ({:}) : axis keyword arguments.
    """
    
    ax2 = ax.twinx()
    
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    
    for x, y, c in xycs:
        ax2.plot(x, y, c = c, alpha = kwargs.get('alpha', 0.5))
        min_x = min(min_x, min(x))
        min_y = min(min_y, min(y))
        max_x = max(max_x, max(x))
        max_y = max(max_y, max(y))
        
    ax2.set_yticks(kwargs.get('yticks', []))
    ax2.set_xlabel(kwargs.get('xlabel', ""))
    ax2.set_ylabel(kwargs.get('ylabel', ""))
    ax2.set_xlim(kwargs.get('xlim', (min_x, max_x)))
    ax2.set_ylim(kwargs.get('ylim', (min_y, max_y)))
    
    
def joint_plot(x, y, 
               w_x = None,
               w_y = None,
               cmap = plt.get_cmap('winter_r'), 
               color = 'springgreen',
               alpha_histo_2d = 0.95, 
               alpha_histo_1d = 0.7,
               **kwargs):
    """
    Creates a seaborn style joint histogram plot.
    Parameters:
        x (np.ndarray[n_observations]) : independent variable plotted on the abscissa
        y (np.ndarray[n_observations]) : independent variable plotted on the ordinate
        cmap (plt.colormap) : colormap of the 2D histogram
        color (str) : color of marginal distributions.
        alpha_histo_2d (float) : transparency of the 2D histogram
        alpha_histo_1d (float) : transparency of the marginal distributions
        kwargs ({:}) : additional axis keywords
    """

    # set up bins
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    bin_x = kwargs.get('xlim', (min_x, max_x))
    bin_y = kwargs.get('ylim', (min_y, max_y))
    
    # create 2-by-2 grid of subplots
    grid_ = gridspec.GridSpec(2, 2, width_ratios = [7, 1], height_ratios = [1, 7])
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    
    # x marginal histo
    ax_x = plt.subplot(grid_[0], 
                      xticks=[], 
                      yticks=[], 
                      frameon = False)
    
    ax_x.hist(x, 
             weights = w_x,
             bins = bin_x[1] - bin_x[0],
             range =  bin_x,
             density = True,
             color = color, 
             rwidth = 0.85,
             alpha = alpha_histo_1d,
             orientation = 'vertical')
     
    # y marginal histo
    ax_y = plt.subplot(grid_[3], 
                      xticks=[], 
                      yticks=[], 
                      frameon = False)
         
    ax_y.set_ylim(bin_y)
    
    ax_y.hist(y, 
              weights = w_y,
              bins = bin_y[1] - bin_y[0],
              range = bin_y,
              density = True,
              rwidth = 0.85,
              color = color, 
              alpha = alpha_histo_1d,
              orientation = 'horizontal')
        
    # 2D histogram
    # calculate
    histo_2d = calculate_histo_2d(x, y, bin_x, bin_y, w_x = w_x, w_y = w_y)
     
    # plot
    ax_xy = plt.subplot(grid_[2], sharex = ax_x, sharey = ax_y)
    cax = ax_xy.scatter(histo_2d[0], 
                        histo_2d[1], 
                        c = histo_2d[2], 
                        cmap = cmap, 
                        alpha = alpha_histo_2d)
    
    # set up limits
    ax_xy.set_xlim(bin_x)
    ax_xy.set_ylim(bin_y)
    
    # set up ticks
    ax_xy.set_xticks(kwargs.get('xticks', []))
    ax_xy.set_yticks(kwargs.get('yticks', []))
    ax_xy.set_xticklabels(kwargs.get('xticklabels', []))
    ax_xy.set_yticklabels(kwargs.get('yticklabels', []))
    
    ax_xy.grid(True)
    ax_xy.set_facecolor('whitesmoke')
    ax_xy.set_xlabel(kwargs.get('xlabel' , ""))
    ax_xy.set_ylabel(kwargs.get('ylabel', ""))
    
    # unset tick labels on marginals
    for tic in ax_x.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    
    for tic in ax_y.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    
    #Bring the marginals closer to the scatter plot
    fig.tight_layout(pad = 1)
    
    return ax_xy, ax_x, ax_y

def plot_pair_grid(data, n_bins = 50, labels = None, sel = None, **kwargs):
    """
    Creates an sns-style density grid of a selection of variables.
    Parameters:
        data ({pandas.DataFrame, np.ndarray}) : underlying data.
            if a numpy ndarray the it has the layout [n_features, n_samples]
            
        n_bins (int) number of bins. Default: 50.
        
        sel (array-like) : labels or indices of features to process:
            if 'data' is a DataFrame sel is the list of column names
            if 'data' is a numpy array sel are the **row** indices
        
        labels (array-like) : list of data labels. Default: None.
        
        kwargs ({:}) : keyword arguments passed to the axis objects.
            it has the format kwargs[<feature_name>][<kwarg_name>] : <kwarg_value>
            Default: None.
    """
    # input checks
    if isinstance(data, pd.DataFrame):
        sel_ = data.columns
        
    elif isinstance(data, np.ndarray):
        
        if len(data.shape) != 2:
            raise ValueError("'data' must be pd.DataFrame or np.ndarray of dimension 2.")
            
        sel_ = np.arange(data.shape[0])
        
    else:
        raise ValueError("'data' must be pd.DataFrame or np.ndarray of dimension 2. Got: {0}".format(type(data)))
        
    if (sel is not None) and (labels is not None):
        if len(sel) != len(labels):
            raise ValueError("'sel' and 'labels' have unequal lengths ({0} != {1})".format(len(sel), len(labels)))
    
    # if no selection has been provided use all columns
    if sel is not None:
        sel_ = sel
        
    # set up labels
    if labels is None:
        labels_ = [str(x) for x in sel_]
    else:
        labels_ = labels

    # create plot
    n_features = len(labels_)
    fig, axes = plt.subplots(n_features, n_features, sharex = True, sharey = True, 
                             gridspec_kw = {'wspace' : 0.02, 'hspace' : 0.02})
    fig.set_size_inches(12, 12)
    axs = axes.flat
        
    # loop over pairs of features
    for c1 in sel_:
        for c2 in sel_:
            ax = next(axs)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if c2 == c1:
                xs = np.arange(n_bins)
                histo, _ = np.histogram(data[c1], bins = n_bins)
                scale_y = 0.85 * n_bins / np.max(histo)
                ax.bar(xs, histo * scale_y, color = 'springgreen', alpha = 0.8)
                continue
                
            histo, _, _ = np.histogram2d(data[c1], data[c2], bins = n_bins)
            ax.contour(histo, cmap = plt.get_cmap('winter_r'), levels = 100)
            
            ax.set_xlim(kwargs.get(c1, {}).get('xlim', None))
            ax.set_ylim(kwargs.get(c1, {}).get('ylim', None))
            
     
    # decorate bottom axes
    for axb, label, sel_name in zip(axes[-1, :], labels_, sel_):    
        axb.set_xlabel(label)

    # decorate left axes
    for axl, label, sel_name in zip(axes[:, 0], labels_, sel_):    
        axl.set_ylabel(label)