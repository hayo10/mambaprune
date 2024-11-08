
num_channels = 50
num_plots = (tmax + 1) // num_channels
num_rows = len(graph_df.index)
cmap = plt.cm.get_cmap('viridis', num_rows)
min_length = 30

l, c = data_array.shape
for idx in range(c // num_channels + 1):
    start_channel = idx * num_channels
    end_channel = start_channel + num_channels

    if end_channel >= c:
        end_channel = c

    plt.figure(figsize=(12, 6))
  
    for channel in range(start_channel, end_channel):
        consecutive_count = 0
        for row in range(l):
            if graph_df.iloc[row, channel] == 1:
                consecutive_count += 1
            else:
                if consecutive_count >= min_length:
                    plt.plot([row - consecutive_count, row], [channel, channel], color=cmap(row / num_rows), linewidth=1)
                consecutive_count = 0
        if consecutive_count >= min_length:
            plt.plot([l - consecutive_count, l], [channel, channel], color=cmap(row / num_rows), linewidth=1)
    
    plt.ylim(start_channel , end_channel)
    plt.xlim(0, l)
    plt.title(f'Channels {start_channel} to {end_channel - 1}')
    plt.xlabel('Rows')
    plt.ylabel('Channels')
    plt.show()
