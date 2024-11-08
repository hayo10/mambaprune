import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('outliers_995.csv', na_values=['NaN'])

value_columns = df.columns[2:]
df['MinValue'] = df[value_columns].min(axis=1)
df['MaxValue'] = df[value_columns].max(axis=1)

dfs_by_name = {name: group.reset_index(drop=True) for name, group in df.groupby('Name')}

dfs_by_name.keys()

df_1 = dfs_by_name['pre_hidden_states']

chs = df_1.iloc[:, 2:-2].apply(lambda row: row.dropna().values, axis=1).to_numpy()
np.set_printoptions(suppress=True)
tmin = int(df_1['MinValue'].min())
tmax = int(df_1['MaxValue'].max())

graph_df = pd.DataFrame({i: [0] * len(chs) for i in range(tmin, tmax + 1)})
for idx, arr in enumerate(chs):
    for ch in arr:
        graph_df.loc[idx, ch] = 1
data_array = graph_df.to_numpy()

num_channels = 500
num_plots = (tmax + 1 - tmin) // num_channels
num_rows = len(graph_df.index)
cmap = plt.cm.get_cmap('viridis', num_rows)
min_length = 30

l, c = data_array.shape
for idx in range(num_plots + 1):
    start_channel = idx * num_channels
    end_channel = start_channel + num_channels

    if end_channel >= c:
        end_channel = c
        print('end channel ', end_channel)

    plt.figure(figsize=(12, 6))
  
    for channel in range(start_channel, end_channel):
        for row in range(l):
            if graph_df.iloc[row, channel] == 1:
                consecutive_count += 1
            else:
                if consecutive_count >= min_length:
                    plt.plot([row - consecutive_count, row], [channel, channel], color=cmap(channel / num_channels / (idx+1)), linewidth=1)
                consecutive_count = 0
        if consecutive_count >= min_length:
            plt.plot([l - consecutive_count, l], [channel, channel], color=cmap(channel / num_channels / (idx+1)), linewidth=1)
    
    plt.ylim(start_channel+tmin , end_channel+tmin)
    plt.xlim(0, l)
    plt.title(f'Channels {start_channel +tmin} to {end_channel + tmin- 1}')
    plt.xlabel('Rows')
    plt.ylabel('Channels')
    plt.show()
