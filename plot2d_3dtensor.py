import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def plot2d(out, x_vals):
    
    percentile_75_upper = np.percentile(out, 75)  
    percentile_99_upper = np.percentile(out, 99) 
    percentile_25_lower = np.percentile(out, 25)  
    percentile_99_lower = np.percentile(out, 1)
    percentile_999_lower = np.percentile(out, 0.1)
    percentile_999_upper = np.percentile(out, 99.9) 

    min_values = np.min(out, axis=0)
    max_values = np.max(out, axis=0)
    
    idx_list = np.array([])
    
    for i in range(out.shape[0]):
        
        plt.plot(x_vals, np.where((percentile_25_lower < out[i]) & (out[i] < percentile_75_upper), out[i], np.nan), color='#FFDC7F', alpha=0.4)
        
        plt.plot(x_vals, np.where((percentile_99_lower <= out[i]) & (out[i] <= percentile_25_lower), out[i], np.nan), color='#B4D6CD', alpha=0.4)
        plt.plot(x_vals, np.where((percentile_75_upper <= out[i]) & (out[i] <= percentile_99_upper), out[i], np.nan), color='#B4D6CD', alpha=0.4)
        
        
        lower_outliers = np.where(percentile_999_lower > out[i])
        upper_outliers = np.where(percentile_999_upper < out[i])
        
        idx_list = np.append(idx_list, lower_outliers)
        idx_list = np.append(idx_list, upper_outliers)


    plt.scatter(x_vals, np.where(min_values < percentile_999_lower, min_values, np.nan), color='#AF47D2', alpha=0.6, s=5)
    plt.scatter(x_vals, np.where(max_values > percentile_999_upper, max_values, np.nan), color='#AF47D2', alpha=0.6, s=5)

    idx_list = np.sort(idx_list)
    
    return idx_list

def plot2d_graph_3d_tensor_activation(output, transposed):
    out = output.detach().cpu().numpy()
    if transposed:
        b, d, l = out.shape
        out = np.transpose(out, (0, 2, 1))
    else:
        b, l, d = out.shape

    channels = np.array([])
    x_vals = np.arange(d)

    range_min = np.min(out)
    range_max = np.max(out)
    
    for i in range(b):
        r_idx_list = plot2d(out[i], x_vals)
        idx_list = list(map(int, r_idx_list))
        channels = np.append(channels, r_idx_list)
    
    unique_ch, counts = np.unique(channels, return_counts=True)
    channels = unique_ch[counts > 1]

    range_min = range_min * 2
    range_max = range_max * 2
    plt.xlabel("channel")
    plt.ylabel("values")
    plt.xlim(0, d)
    plt.ylim(range_min, range_max)
    plt.title("Activation Values ")

    plt.show()
    
    return idx_list
