import pandas as pd
import numpy as np


def window_feature_extract(data, column, window_size):
    features_df = pd.DataFrame(columns=['max_' + column, 'min_' + column, 'skew_' + column, 'std_' + column, \
                                        'mean_' + column, 'median_' + column, 'variance_' + column, 'range_' + column, \
                                        'energy_' + column, 'rms_' + column])

    # Calculate the number of windows based on the window size and the sampling rate
    num_windows = int(len(data) / (window_size * 5))

    # Iterate over each window and calculate features
    for i in range(num_windows):
        start_index = i * window_size * 5
        end_index = start_index + window_size * 5

        window_data = data[start_index:end_index]
        f_max = window_data[column].max()
        f_min = window_data[column].min()
        skew = window_data[column].skew()
        std = window_data[column].std()
        mean = window_data[column].mean()
        median = window_data[column].median()
        variance = window_data[column].var()
        f_range = f_max - f_min
        energy = np.sum(window_data[column] ** 2)
        rms = np.sqrt(np.mean(window_data[column] ** 2))

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
        features_df = pd.concat([features_df, pd.DataFrame([new_row])], ignore_index=True)

    return features_df

# Moving average filter
def SMA(data, columns, window_size):
    smoothed_data = data.copy()
    for col in columns:
        # Use rolling window to calculate the moving average, then fill NaNs at the start with original values
        smoothed_col = data[col].rolling(window=window_size, min_periods=1).mean()
        smoothed_data[col] = smoothed_col

    return smoothed_data

