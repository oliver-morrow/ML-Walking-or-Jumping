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

def window_feature_extract(window_list, columns):
    features_df = pd.DataFrame()
    for column in columns:
        # Create a new DataFrame to store the calculated features for the current column
        column_features_df = pd.DataFrame(columns=['max_' + column, 'min_' + column, 'skew_' + column, 'std_' + column, \
                                                'mean_' + column, 'median_' + column, 'variance_' + column, 'range_' + column, \
                                                'energy_' + column, 'rms_' + column, ])

        # Iterate over each window in window_list and calculate features
        for i in range(len(window_list)):
            # Calculate the maximum value of the current column in the window
            f_max = window_list[i][column].max()
            # Calculate the minimum value of the current column in the window
            f_min = window_list[i][column].min()
            # Calculate the skewness of the current column in the window
            skew = window_list[i][column].skew()
            # Calculate the standard deviation of the current column in the window
            std = window_list[i][column].std()
            # Calculate the mean of the current column in the window
            mean = window_list[i][column].mean()
            # Calculate the median of the current column in the window
            median = window_list[i][column].median()
            # Calculate the variance of the current column in the window
            variance = window_list[i][column].var()
            # Calculate the range of the current column in the window
            f_range = f_max - f_min
            # Calculate the energy of the current column in the window
            energy = np.sum(window_list[i][column] ** 2)
            # Calculate the root mean square of the current column in the window
            rms = np.sqrt(np.mean(window_list[i][column] ** 2))

            # Append the calculated features for the current window to the DataFrame
            new_row = {'max_' + column: f_max,
                       'min_' + column: f_min,
                       'skew_' + column: skew,
                       'std_' + column: std,
                       'mean_' + column: mean,
                       'median_' + column: median,
                       'variance_' + column: variance,
                       'range_' + column: f_range,
                       'energy_' + column: energy,
                       'rms_' + column: rms}
            column_features_df = pd.concat([column_features_df, pd.DataFrame([new_row])], ignore_index=True)

        # Concatenate the features DataFrame for the current column with the overall features DataFrame
        features_df = pd.concat([features_df, column_features_df], axis=1)

    return features_df

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