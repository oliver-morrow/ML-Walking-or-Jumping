import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# FUNCTION to label data
def label_data(data_fp, label):
    # read into dataframe
    df = pd.read_csv(data_fp)
    # add label column with label
    df['label'] = label

    return df


# READING CSV files into data frames (replace the placeholders with your actual file paths)
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

# SHUFFLE data frames in 5 second inteverals
def shuffle_data(df, time = 'Time (s)', window_size=5):

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

    # shuffle the windows
    windows_shuffled = np.random.shuffle(windows)

    # concatenate into a single dataframe
    shuffled_df = pd.concat(windows_shuffled)

    return shuffled_df


# shuffle the data
oliver_walking_shuffled = shuffle_data(oliver_walking_labeled)
matthew_walking_shuffled = shuffle_data(matthew_walking_labeled)
daniel_walking_shuffled = shuffle_data(daniel_walking_labeled)

oliver_jumping_shuffled = shuffle_data(oliver_jumping_labeled)
matthew_jumping_shuffled = shuffle_data(matthew_jumping_labeled)
daniel_jumping_shuffled = shuffle_data(daniel_jumping_labeled)

# combine dataframes by activity
walking_df = pd.concat([oliver_walking_shuffled, matthew_walking_shuffled, daniel_walking_shuffled], ignore_index=True)
jumping_df = pd.concat([oliver_jumping_shuffled, matthew_jumping_shuffled, daniel_jumping_shuffled], ignore_index=True)

# SPLIT data into training and testing sets
walking_train, walking_test = train_test_split(walking_df, test_size=0.1)
jumping_train, jumping_test = train_test_split(jumping_df, test_size=0.1)

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

# PLOT data for walking and jumping, all three axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Plotting walking data
walking_data = walking_df[['Time (s)', 'X', 'Y', 'Z']]
walking_data.plot(x='Time (s)', y='X', ax=axes[0, 0], title='Walking - X axis')
walking_data.plot(x='Time (s)', y='Y', ax=axes[0, 1], title='Walking - Y axis')
walking_data.plot(x='Time (s)', y='Z', ax=axes[0, 2], title='Walking - Z axis')

# Plotting jumping data
jumping_data = jumping_df[['Time (s)', 'X', 'Y', 'Z']]
jumping_data.plot(x='Time (s)', y='X', ax=axes[1, 0], title='Jumping - X axis')
jumping_data.plot(x='Time (s)', y='Y', ax=axes[1, 1], title='Jumping - Y axis')
jumping_data.plot(x='Time (s)', y='Z', ax=axes[1, 2], title='Jumping - Z axis')

plt.tight_layout()
plt.show()

############################################################################################################

# Moving average filter
def moving_average_filter(data, window_size=5):
    return data.rolling(window=window_size).mean()

############################################################################################################

'''
From each time window (the 5-second segments that you created and stored in the HDF5 file),
extract a minimum of 10 different features. These features could be maximum, minimum, range,
mean, median, variance, skewness, etc. Additional features may be explored as well. After feature
extraction has been performed, you will be required to apply a normalization technique for
preventing features with larger scales from disproportionately influencing the results. Common
normalization techniques are min-max scaling, z-score standardization, etc.
'''
# FEATURE Extraction AND NORMALIZATION
features = pd.DataFrame(columns=['mean','std','max',
                                 'min','median','skew',
                                 'kurtosis','energy','rms', 
                                 'range', 'variance'])
window_size = 5
features['mean'] = walking_df.rolling(window=window_size).mean()
features['std'] = walking_df.rolling(window=window_size).std()
features['max'] = walking_df.rolling(window=window_size).max()
features['min'] = walking_df.rolling(window=window_size).min()
features['median'] = walking_df.rolling(window=window_size).median()
features['skew'] = walking_df.rolling(window=window_size).skew()
features['kurtosis'] = walking_df.rolling(window=window_size).kurtosis()
features['energy'] = walking_df.rolling(window=window_size).apply(lambda x: np.sum(x**2))
features['rms'] = walking_df.rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))

features['mean'] = jumping_df.rolling(window=window_size).mean()
features['std'] = jumping_df.rolling(window=window_size).std()
features['max'] = jumping_df.rolling(window=window_size).max()
features['min'] = jumping_df.rolling(window=window_size).min()
features['median'] = jumping_df.rolling(window=window_size).median()
features['skew'] = jumping_df.rolling(window=window_size).skew()
features['kurtosis'] = jumping_df.rolling(window=window_size).kurtosis()
features['energy'] = jumping_df.rolling(window=window_size).apply(lambda x: np.sum(x**2))
features['rms'] = jumping_df.rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))

newfeatures = features.dropna()
print(newfeatures)

# NORMALIZATION
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(newfeatures)
print(normalized_features)

'''
Using the features from the preprocessed training set, train a logistic regression model to classify
the data into 'walking' and 'jumping' classes. Once training is complete, apply it on the test set
and record the accuracy. You should also monitor and record the training curves during the
training process. Note that during the training phase, your test set must not leak into the training
set (no overlap between the segments used for training and testing).
'''
# TRAINING logistic regression model
X_train = normalized_features
y_train = np.concatenate((np.zeros(len(walking_train)), np.ones(len(jumping_train))))

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# TESTING logistic regression model
X_test = scaler.transform(features.dropna())
y_test = np.concatenate((np.zeros(len(walking_test)), np.ones(len(jumping_test))))

# Predict the classes for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model to a joblib file
joblib.dump(model, 'logistic_regression_model.joblib')
