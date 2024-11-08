import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import pandas as pd

def plot_activation(output):
    
    ## 여러개 output 반환 시
    if isinstance(output, tuple):
        out = output[0].detach().cpu().numpy()
    else:
        out = output.detach().cpu().numpy()
    
    range_min = out.min()
    range_max = out.max()
    x_vals = np.arange(out.shape[-1])
    median = np.median(out)
    
    out = out.reshape(-1, out.shape[-1])
    l, d = out.shape

      
    for i in range(out.shape[0]):
        percentile_75_upper = np.percentile(out[i], 75)  # 중앙값 위쪽에서 75%
        percentile_99_upper = np.percentile(out[i], 99) # 중앙값 위쪽에서 99%
        percentile_75_lower = np.percentile(out[i], 25)  # 중앙값 아래쪽에서 25% (즉, 75%)
        percentile_99_lower = np.percentile(out[i], 1)

        
        plt.plot(x_vals, np.where((percentile_75_lower < out[i]) & (out[i] < percentile_75_upper), out[i], np.nan), color='#FFDC7F', alpha=0.4)
        
        plt.plot(x_vals, np.where((percentile_99_lower <= out[i]) & (out[i] <= percentile_75_lower), out[i], np.nan), color='#B4D6CD', alpha=0.4)
        plt.plot(x_vals, np.where((percentile_75_upper <= out[i]) & (out[i] <= percentile_99_upper), out[i], np.nan), color='#B4D6CD', alpha=0.4)
        
        plt.plot(x_vals, np.where(out[i] <= percentile_99_lower, out[i], np.nan), color='#AF47D2', alpha=0.6)
        plt.plot(x_vals, np.where(out[i] >= percentile_99_upper, out[i], np.nan), color='#AF47D2', alpha=0.6)
        
    plt.xlabel("channel")
    plt.ylabel("values")
    plt.xlim(0, out.shape[-1])
    plt.ylim(range_min, range_max)
    plt.title("Activation Values ")
    plt.show()
    
    df = pd.DataFrame({
    "Minimum": [range_min],
    "Median": [median],
    "Maximum": [range_max],
    "90프로 upper value": [percentile_99_upper],
    "90프로 lower value": [percentile_99_lower]
    })
    
    print(df)