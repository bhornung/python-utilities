import numpy as np
from scipy.sparse import coo_matrix


def calculate_histo_2d(x, y, bin_x, bin_y, w_x = None, w_y = None):
    """
    Calculates the 2D histogram of two sets of related observations i.e. P(x, y)
    Parameters:
        x (np.ndarray) : first set of observatios
        y (np.ndarray) : second set of observations
        w (np.ndarray) : weights of x. w_x **must be** aligned with the x and y.
    Returns:
        histo (np.ndarray[3, :]) : coordinate format histogram. (x, y, count(x,y))).
    """
  
    # make neat bins
    bin_x_range = np.arange(bin_x[0], bin_x[1] + 1)
    bin_y_range = np.arange(bin_y[0], bin_y[1] + 1)
        
    # weights
    if w_x is None and w_y is None:
        w = None
    elif w_x is None:
        w = w_y
    elif w_y is None:
        w = w_x
    else:
        w = w_x * w_y
        
    histo, _, _ = np.histogram2d(x, y, 
                               bins = [bin_x_range, bin_y_range],
                               weights = w)
    
    # reduce to x, y, z format
    histo_sparse = coo_matrix(histo)
    x_y_cnt = histo = np.stack([histo_sparse.row + bin_x[0], 
                                histo_sparse.col + bin_y[0], 
                                histo_sparse.data])
    
    return histo


def calc_probability_from_coo_histo(histo, x = None, y = None, exclude_values = None):
    """
    Calculates conditional probavility from a coo format histogram.
    Parameters:
        histo (np.ndarray[3, n_observations]) : x, y, p(x, y) histogram
        x ({np.float, np.int, None}) : slice along this value of x --> P(y | X = x). Deafult None.
        y ({np.float, np.int, None}) : slice along this value of y --> P(x | Y = y). Default None.
        exclude_values ({array-like, None}) : values to be excluded from the dimension which is not selected. 
            Default: None.
    
    Returns:
        p_cond (np.ndarray[2, ]) the conditional probabilites
        1st row: coordinates
        2nd row probabilities
    """
    
    if len(histo.shape) != 2:
        raise ValueError("'histo' must be a 2D array")
        
    if histo.shape[0] != 3:
        raise ValueError("'histo' first dimension must be 2 (expecting coo format histogram)")
    
    if (x is None) and (y is None):
        raise ValueError("Either 'x' or 'y' must be specified")
        
    if (x is not None) and (y is not None):
        raise ValueError("Only 'x' xor 'y' can be specified")
        
    # select row or coordinate (x or y)
    i_sel = (0, 1)[x is None]
    j_sel = (1, 0)[x is None]
    v_sel = (x, y)[x is None]
    
    # select coordinates and values
    keep_mask = histo[i_sel] == v_sel  
    coords = histo[j_sel, keep_mask]
    vals = histo[2, keep_mask]
     
    # shortcut on no find
    if np.all(~keep_mask):
        return np.array([[],[]])
    
    # calculate total count
    norm = vals.sum()
    
    # exclude values
    if exclude_values is not None:
        
        keep_mask = np.full(vals.shape, True)
        for e_val in exclude_values:
            keep_mask &= coords != e_val
        vals = vals[keep_mask]
        coords = coords[keep_mask]
    
    # calculate probability with correct norm
    probs = calculate_scaled_probability(vals, norm = norm)
    
    p_cond = np.stack([coords, probs])
    
    return p_cond


def calc_average_probability_from_coo_histo(histo, xs = None, ys = None, exclude_values = None):
    """
    Calculates conditional probavility from a coo format histogram.
    Parameters:
        histo (np.ndarray[3, n_observations]) : x, y, p(x, y) histogram
        xs ({array-like, None}) : slice and average along this value of x --> P(y | X = x). Deafult None.
        ys ({array-like, None}) : slice and average along this value of y --> P(x | Y = y). Default None.
        exclude_values ({array-like, None}) : values to be excluded from the dimension which is not selected. 
            Default: None.
    
    Returns:
        p_cond (np.ndarray[2, ]) the conditional probabilites
        1st row: coordinates
        2nd row probabilities
    """
    
    from collections import defaultdict
    
    if len(histo.shape) != 2:
        raise ValueError("'histo' must be a 2D array")
        
    if histo.shape[0] != 3:
        raise ValueError("'histo' first dimension must be 2 (expecting coo format histogram)")
    
    if (xs is None) and (ys is None):
        raise ValueError("Either 'x' or 'y' must be specified")
        
    if (xs is not None) and (ys is not None):
        raise ValueError("Only 'x' xor 'y' can be specified")
        
    # select row or coordinate (x or y)
    i_sel = (0, 1)[xs is None]
    j_sel = (1, 0)[xs is None]
    v_sels = (xs, ys)[xs is None]
    
    norms = []
    p_conds = []
    dict_ = defaultdict(int)
    
    for v_sel in v_sels:
        # select coordinates and values
        keep_mask = histo[i_sel] == v_sel  
        coords = histo[j_sel, keep_mask]
        vals = histo[2, keep_mask]
         
        # shortcut on no find
        if np.all(~keep_mask):
            return np.array([[],[]])
        
        # calculate total count
        norm = vals.sum()
        
        # exclude values
        if exclude_values is not None:
            
            keep_mask = np.full(vals.shape, True)
            for e_val in exclude_values:
                keep_mask &= coords != e_val
            vals = vals[keep_mask]
            coords = coords[keep_mask]
        
        # calculate probability with correct norm
        probs = calculate_scaled_probability(vals, norm = norm)
        
        p_conds.append(np.stack([coords, probs]))
        norms.append(norm)
        
    # merge probs I'm know it is ugly
    for norm, p_cond in zip(norms, p_conds):
        for c, p in p_cond.T:
            dict_[c] += p * norm
            
    # reshape
    p_cond = np.array([[c, p] for c, p in sorted(dict_.items())]).T
    # normalise
    p_cond[1] /= sum(norms)
    
    return p_cond


def calculate_scaled_probability(X, norm = None):
    """
    Calculates a scaled probability.
    Parameters:
        X (np.ndarray(n_observations)) : event counts
        norm ({np.float, None}) : norm factor. The probability is normalised by X.sum() / norm. Default: None.
        
    Returns:
        p (np.ndarray[n_observations]) : scaled probability
    """
    
    x_sum = X.sum()
    norm_ = (norm, x_sum)[norm is None]
    
    p = X / x_sum * (x_sum / norm_)
    
    return p


def collector(xs, ws, ys):
    """
    Collects statistics about the time series.
    Parameters:
        xs (iterable) : time series values
        ys (iterable) : year (or other grouper) for the time series
        ww (iterable) : weight of each point
        
    Returns:
        x_stats ({}) : statistics keyed by the individual x values.
            first : year of first appearance
            last : year of last appearance
            pos : sum of positions (to calculate mean later on)
            cnt : overall count
            f_pos : sum of positions in the first year
            f_cnt : count in first half year
    """
    
    x_stats = {}
    
    for idx, (x, w, y) in enumerate(zip(xs, ws, ys)):
        
        pos = idx % 100 * w
        
        if not x in x_stats:
            
            dict_ = {'first' : y, 
                     'last' : y,
                     'pos' : pos,
                     'cnt' : 1,
                     'pos_first' : pos,
                     'cnt_first' : 1}
            
            x_stats.update({x : dict_})
            
        else:
            
            x_stats[x]['last'] = y
            x_stats[x]['pos'] += pos
            x_stats[x]['cnt'] += 1
            
            # first year performance
            if y == x_stats[x]['first']:
                x_stats[x]['pos_first'] += pos
                x_stats[x]['cnt_first'] += 1
                                
    return x_stats


def _calc_least_diverse_distribution(M, N):
    """
    Calculates the probability distribution of N elements
    arranged in the least uniform manner among M bins. 
    Each bin is required to contain at least one element.
    Parameters:
        M (int) : number of bins
        N (int) : number of elements
    Returns:
        p (np.ndarray) : probability density function of the least uniform distribution.
    """
    
    p = np.ones(M, dtype = np.float)
    p[M - 1] = (N - (M - 1)) 
    
    p /= p.sum()
    
    return p


def _calc_most_diverse_distribution(M, N):
    """
    Calculates the probability distribution of N elements
    arranged in the most uniform manner among M bins.
    Parameters:
        M (int) : number of bins
        N (int) : number of elements
    Returns:
        p (np.ndarray) : probability density function of the most uniform distribution.
    """
    
    p = np.full(M, np.floor_divide(N, M), dtype = np.float)
    
    n = M - np.mod(N, M)
    
    if n > 0:
        p[n:] += 1
     
    p /= p.sum()
    
    return p


def _calculate_sort_counts(X):
    """
    Calculates and sorts the unique elements of a sample.
    Parameters:
        X (np.ndarray) : observations
    Returns:
        n_unique_cnts (np.ndarray of int) : the counts of unique elements in ascending order.
    """

    _, n_unique_cnts = np.unique(X, return_counts = True)
    
    n_unique_cnts = np.sort(n_unique_cnts)
    
    return n_unique_cnts
	
    
def calculate_adjusted_entropy(X):
    """
    Calculates the adjusted entropy of a sample.
    Corrects for nonzero appearances. The reference worst and best best distributions.
    Parameters:
        X (np.ndarray) : observations e.g. [1,2,3,1,1,6,6,1,1,1,1]
    Returns:
        h_adj (float) : adjusted entropy
    """

    counts = _calculate_sort_counts(X[~np.isnan(X)])
     
    M = counts.size
    N = counts.sum()
    p = counts / N
    
    p_worst = calc_least_diverse_distribution(M, N)
    p_best = calc_most_diverse_distribution(M, N)
    
    h = calculate_entropy(p)
    h_worst = calculate_entropy(p_worst)
    h_best = calculate_entropy(p_best)
    
    h_adj = (h - h_worst) / ( h_best - h_worst)
    
    return h_adj
	
	
def calculate_adjusted_gini(X):
    """
    Calculates the adjusted Gini coefficient.
    Corrects for nonzero appearances. The reference worst and best best distributions
    are used as opposed to the y = 0 and y = x.
    Parameters:
        X (np.ndarray) : observations e.g. [1,2,3,1,1,6,6,1,1,1,1]
    Returns:
        rho_g (float) : adjusted Gini coefficient
    """
     
    counts = _calculate_sort_counts(X[~np.isnan(X)])
     
    M = counts.size
    N = counts.sum()
    p = counts / N
    
    p_worst = _calc_least_diverse_distribution(M, N)
    p_best = _calc_most_diverse_distribution(M, N)
    
    l = np.cumsum(p)
    l_worst = np.cumsum(p_worst)
    l_best = np.cumsum(p_best)
    rho_g = 1 - (l_best - l).sum() / (l_best - l_worst).sum()
    
    return rho_g


def calculate_adjusted_density(X):
    """
    Calculates the unique objects in a collection and the size of the collection.
    Corrects for nonzero appearances.
    density = (number of unique elements / number of all elements)
    Parameters:
        X (np.ndarray) : observations e.g. [1,2,3,1,1,6,6,1,1,1,1]
    Returns:
        rho_adj (float) : adjusted density
    """
    
    cnts = _calculate_sort_counts(X[~np.isnan(X)])
    
    n_p = cnts.sum()
    n_l = cnts.size
    
    rho = n_l / n_p
    
    rho_adj = (rho - 1 / n_p) / (1 - 1 / n_p)
    
    return rho_adj


def calculate_entropy(p):
    """
    Calculates the entropy of a ditributio in shannons.
    Parameters:
        p (np.ndarray) : normalised probability distribution
        
    Returns:
        entropy (np.ndarray) : entropy of the distribution
    """
    
    entropy = - np.dot(p, np.log2(p))
    
    return entropy

    
def calculate_gini_simpson(X):
    """
    Calculates the Gini--Simpson coefficient of a sample.
    Parameters:
        X (np.ndarray) : observations e.g. [1,2,3,1,1,6,6,1,1,1,1]
    Returns:
        gini_simpson (float) : Gini--Simpson coefficient
    """
    
    counts = _calculate_sort_counts(X[~np.isnan(X)])
    N = counts.sum()
    p = counts / N
    
    gini_simpson = 1 - np.dot(p, p)
    
    return gini_simpson
	
	
def omit_value(value):
    """
    Wrapper to facilitate omission of specified values from an array.
    E.g. one wishes to exclude the value 9 from an array passed to func.
    Then define func as
    @omit_value(9):
    def func(X):
        # process X
    """
    def omit(func):
        def func_wrapper(X):
            res = func(X[X != value])
            return res
    
        return func_wrapper
    return omit	
	