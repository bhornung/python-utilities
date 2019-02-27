import numpy as np
import pandas as pd


	
def calculate_rolling_mean_std(df_, win_size, 
                               pad_size = None, 
                               add_expanding = False):
    """
    Calculates the rolling mean and \pm 1 standard deviations of a series.
    Parameters:
        df_ ({pd.DataFrame, pd.Series}) : series to process
        win_size (int) : window size
        pad_size (int) : prepend the rolling mean with given number of np.nans. Default: None.
        add_expanding (bool) : whether to calculate the rolling means initial np.nan entries 
            with the expanding mean. Default: False.
     Returns:
        x (pd.DataFrame) : columns = ['mean', 'mean-std', 'mean+std']. 
            The rolling mean, mean - std, mean + std of the series.
    """
    
    mean_ = df_.rolling(win_size).mean()
    std_ = df_.rolling(win_size).std()
    
    x = pd.DataFrame() 
    x['mean'] = mean_
    x['mean-std'] = mean_ - std_
    x['mean+std'] = mean_ + std_
     
    # fill up initial nan's with expanding stats
    if add_expanding:
        
        x_expanding = pd.DataFrame()
        
        mean_e_ = df_[:win_size].expanding(2).mean()
        std_e_ = df_[:win_size].expanding(2).std()
        
        x.loc[: win_size, 'mean'] = mean_e_
        x.loc[: win_size, 'mean-std'] = x.loc[: win_size, 'mean'] - std_e_
        x.loc[: win_size, 'mean+std'] = x.loc[: win_size, 'mean'] + std_e_
        
    if pad_size is None:
        return x

    if pad_size is not None:
        x_padding = pd.DataFrame(np.full((pad_size, 3), np.nan),
                               columns = ['mean', 'mean-std', 'mean+std'])
    
        return pd.concat([x_padding, x])