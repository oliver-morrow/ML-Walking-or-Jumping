import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib

############################################################################################################
# FUNCTION to label data
def label_data(data_fp, label):
    # read into dataframe
    df = pd.read_csv(data_fp)
    # add label column with label
    df['label'] = label

    return df

############################################################################################################
# READING CSV files into data frames
file_paths = {
    'Oliver': ['meta/oliver_walking.csv', 'meta/oliver_jumping.csv'],
    'Matthew': ['meta/matthew_walking.csv', 'meta/matthew_jumping.csv'],
    'Daniel': ['meta/daniel_walking.csv', 'meta/daniel_jumping.csv'],
}
oliver_walking_labeled = label_data(file_paths['Oliver'][0], 0.0)
matthew_walking_labeled = label_data(file_paths['Matthew'][0], 0.0)
daniel_walking_labeled = label_data(file_paths['Daniel'][0], 0.0)

oliver_jumping_labeled = label_data(file_paths['Oliver'][1], 1.0)
matthew_jumping_labeled = label_data(file_paths['Matthew'][1], 1.0)
daniel_jumping_labeled = label_data(file_paths['Daniel'][1], 1.0)

############################################################################################################
# SHUFFLE data frames in 5 second inteverals
def shuffle_data(df, time='Time (s)', window_size=5):
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

    # shuffle the windows IN PLACE
    np.random.shuffle(windows)

    # concatenate into a single dataframe
    shuffled_df = pd.concat(windows)

    return shuffled_df

############################################################################################################
# shuffle the data
oliver_walking_shuffled = shuffle_data(oliver_walking_labeled)
matthew_walking_shuffled = shuffle_data(matthew_walking_labeled)
daniel_walking_shuffled = shuffle_data(daniel_walking_labeled)

oliver_jumping_shuffled = shuffle_data(oliver_jumping_labeled)
matthew_jumping_shuffled = shuffle_data(matthew_jumping_labeled)
daniel_jumping_shuffled = shuffle_data(daniel_jumping_labeled)

############################################################################################################
# combine dataframes by activity
walking_df = pd.concat([oliver_walking_shuffled, matthew_walking_shuffled, daniel_walking_shuffled], ignore_index=True)
jumping_df = pd.concat([oliver_jumping_shuffled, matthew_jumping_shuffled, daniel_jumping_shuffled], ignore_index=True)

############################################################################################################
# SPLIT data into training and testing sets
walking_train, walking_test = train_test_split(walking_df, test_size=0.1)
jumping_train, jumping_test = train_test_split(jumping_df, test_size=0.1)

############################################################################################################
# SAVE data to HDF5 file
with h5py.File('data.h5', 'w') as f:
    G1 = f.create_group('/oliver')
    G1.create_dataset('walking', data=oliver_walking_labeled)
    G1.create_dataset('jumping', data=oliver_jumping_labeled)
    
    G2 = f.create_group('/matthew')
    G2.create_dataset('walking', data=matthew_walking_labeled)
    G2.create_dataset('jumping', data=matthew_jumping_labeled)

    G3 = f.create_group('/daniel')
    G3.create_dataset('walking', data=daniel_walking_labeled)
    G3.create_dataset('jumping', data=daniel_jumping_labeled)

    G41 = f.create_group('/dataset/train')
    G41.create_dataset('walking_train', data=walking_train)
    G41.create_dataset('jumping_train', data=jumping_train)

    G42 = f.create_group('/dataset/test')
    G42.create_dataset('walking_test', data=walking_test)
    G42.create_dataset('jumping_test', data=jumping_test)

############################################################################################################
# Moving average filter
def moving_average_filter(data, window_size=5):
    return data.rolling(window=window_size).mean()

############################################################################################################
# FEATURE EXTRACTION & NORMALIZATION
# Define the window size for rolling calculations
window_size = 5

# Create a DataFrame to hold the features
features = pd.DataFrame()

# Calculate rolling features for 'Absolute acceleration (m/s^2)'
features['mean'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).mean()
features['std'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).std()
features['max'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).max()
features['min'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).min()
features['median'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).median()
features['skew'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).skew()
# features['kurtosis'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).kurtosis()
features['variance'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).var().dropna()

features['energy'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sum(x**2), raw=True)

features['rms'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

# Calculate the range as max - min
features['range'] = features['max'] - features['min']

# Drop NaN values from the DataFrame
features = features.dropna().reset_index(drop=True)

# Now print the features DataFrame
print(features)

# Calculate rolling features for 'Absolute acceleration (m/s^2)'
features['mean'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).mean()
features['std'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).std()
features['max'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).max()
features['min'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).min()
features['median'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).median()
features['skew'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).skew()
# features['kurtosis'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).kurtosis()
features['variance'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).var().dropna()

features['energy'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sum(x**2), raw=True)

features['rms'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

# Calculate the range as max - min
features['range'] = features['max'] - features['min']

# Drop NaN values from the DataFrame
features = features.dropna().reset_index(drop=True)

# Now print the features DataFrame
print(features)

# NORMALIZATION
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features.dropna())
print(normalized_features)

###########################################################################################################
# TRAINING logistic regression model
train_data = pd.concat([walking_train, jumping_train])
test_data = pd.concat([walking_test, jumping_test])

# Separate features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Now the model is trained and you can use it to make predictions
predictions = model.predict(X_test)

# You can also measure the accuracy of your model
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

###########################################################################################################
# PLOT data for walking and jumping, all three axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Plotting walking data
walking_data_sorted = walking_df.sort_values(by='Time (s)')
# walking_data_sorted = walking_df[['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']]
walking_data_sorted.plot(x='Time (s)', y='Acceleration x (m/s^2)', ax=axes[0, 0], title='Walking - X acceleration')
walking_data_sorted.plot(x='Time (s)', y='Acceleration y (m/s^2)', ax=axes[0, 1], title='Walking - Y acceleration')
walking_data_sorted.plot(x='Time (s)', y='Acceleration z (m/s^2)', ax=axes[0, 2], title='Walking - Z acceleration')

# Plotting jumping data
jumping_data_sorted = jumping_df.sort_values(by='Time (s)')
# jumping_data = jumping_df[['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']]
jumping_data_sorted.plot(x='Time (s)', y='Acceleration x (m/s^2)', ax=axes[1, 0], title='Jumping - X acceleration')
jumping_data_sorted.plot(x='Time (s)', y='Acceleration y (m/s^2)', ax=axes[1, 1], title='Jumping - Y acceleration')
jumping_data_sorted.plot(x='Time (s)', y='Acceleration z (m/s^2)', ax=axes[1, 2], title='Jumping - Z acceleration')

plt.tight_layout()
plt.show()