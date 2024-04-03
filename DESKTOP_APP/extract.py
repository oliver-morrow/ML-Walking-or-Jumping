import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SPLIT data frames in 5 second inteverals
def split_data(df, time='Time (s)', window_size=5):
    # list to hold individual 5 second windows
    windows = []
    # initialize window time variables
    start_time = 0
    end_time = start_time + window_size

    # split the data into windows
    while end_time <= df[time].max():
        window = df[(df[time] >= start_time) & (df[time] < end_time)]
        windows.append(window)
        start_time = end_time
        end_time += window_size

    return windows

############################################################################################################

def SMA(data, columns, window_size):
    smoothed_data = data.copy()
    for col in columns:
        # Use rolling window to calculate the moving average, then fill NaNs at the start with original values
        smoothed_col = data[col].rolling(window=window_size, min_periods=1).mean()
        smoothed_data[col] = smoothed_col

    return smoothed_data

############################################################################################################

def window_feature_extract(window_list, Column):
    features_df = pd.DataFrame(columns=['max_' + Column, 'min_' + Column, 'skew_' + Column, 'std_' + Column, \
                                        'mean_' + Column, 'median_' + Column, 'variance_' + Column, 'range_' + Column, \
                                        'energy_' + Column, 'rms_' + Column, ])

    # Iterate over each window in walking_data_list and calculate features
    for i in range(len(window_list)):
        f_max = window_list[i][Column].max()
        f_min = window_list[i][Column].min()
        skew = window_list[i][Column].skew()
        std = window_list[i][Column].std()
        mean = window_list[i][Column].mean()
        median = window_list[i][Column].median()
        variance = window_list[i][Column].var()
        f_range = f_max - f_min
        energy = np.sum(window_list[i][Column] ** 2)
        rms = np.sqrt(np.mean(window_list[i][Column] ** 2))

        # Append the calculated features for the current window to the DataFrame
        new_row = {'max_' + Column: f_max,
                   'min_' + Column: f_min,
                   'skew_' + Column: skew,
                   'std_' + Column: std,
                   'mean_' + Column: mean,
                   'median_' + Column: median,
                   'variance_' + Column: variance,
                   'range_' + Column: f_range,
                   'energy_' + Column: energy,
                   'rms_' + Column: rms}
        features_df.dropna()
        features_df = pd.concat([features_df, pd.DataFrame([new_row])], ignore_index=True)

    return features_df  # Corrected indentation for the return statement

############################################################################################################

def plot_features(features_df, title="Features Plot", show_plots=True):
    fig, ax = plt.subplots(figsize=(20, 6))
    
    for col in features_df.columns:
        ax.plot(features_df.index, features_df[col], label=col)
    
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    if show_plots:
        plt.show()
    else:
        return fig

############################################################################################################

def plot_raw_data(df, columns):
    plt.figure(figsize=(20, 6))
    for column in columns:
        plt.plot(df['Time (s)'], df[column], label=column)
    plt.title('Raw Sensor Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Sensor Readings')
    plt.legend()
    plt.show()