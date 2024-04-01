import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score

# READING CSV files into data frames
file_paths = {
    'Oliver': ['meta/oliver_walking.csv', 'meta/oliver_jumping.csv'],
    'Matthew': ['meta/matthew_walking.csv', 'meta/matthew_jumping.csv'],
    'Daniel': ['meta/daniel_walking.csv', 'meta/daniel_jumping.csv'],
}

############################################################################################################
# FUNCTION to label data
def label_data(data_fp, label):
    # read into dataframe
    df = pd.read_csv(data_fp)
    # add label column with label
    df['label'] = label

    return df

oliver_walking_labeled = label_data(file_paths['Oliver'][0], 0.0)
matthew_walking_labeled = label_data(file_paths['Matthew'][0], 0.0)
daniel_walking_labeled = label_data(file_paths['Daniel'][0], 0.0)

oliver_jumping_labeled = label_data(file_paths['Oliver'][1], 1.0)
matthew_jumping_labeled = label_data(file_paths['Matthew'][1], 1.0)
daniel_jumping_labeled = label_data(file_paths['Daniel'][1], 1.0)

############################################################################################################
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

    # concatenate into a single dataframe
    split_df = pd.concat(windows)

    return split_df

############################################################################################################

oliver_walking_split = shuffle(split_data(oliver_walking_labeled))
matthew_walking_split = shuffle(split_data(matthew_walking_labeled))
daniel_walking_split = shuffle(split_data(daniel_walking_labeled))

oliver_jumping_split = shuffle(split_data(oliver_jumping_labeled))
matthew_jumping_split = shuffle(split_data(matthew_jumping_labeled))
daniel_jumping_split = shuffle(split_data(daniel_jumping_labeled))

############################################################################################################
# combine dataframes by activity

walking_df = pd.concat([oliver_walking_split, matthew_walking_split, daniel_walking_split], ignore_index=True)
jumping_df = pd.concat([oliver_jumping_split, matthew_jumping_split, daniel_jumping_split], ignore_index=True)


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
    data = data.iloc[:, 1:-1]
    sma = data.rolling(window=window_size).mean()
    sma = sma.dropna()
    return sma

############################################################################################################
# FEATURE EXTRACTION & NORMALIZATION

features_walk = walking_df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']]
features_jump = jumping_df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']]
# Define the window size for rolling calculations
# Create a DataFrame to hold the features
# Function to calculate features for a specific column
def calculate_features(df, column_name, window_size=5):
    features = pd.DataFrame()
    features['mean_' + column_name] = df[column_name].rolling(window=window_size).mean()
    features['std_' + column_name] = df[column_name].rolling(window=window_size).std()
    features['max_' + column_name] = df[column_name].rolling(window=window_size).max()
    features['min_' + column_name] = df[column_name].rolling(window=window_size).min()
    features['median_' + column_name] = df[column_name].rolling(window=window_size).median()
    features['skew_' + column_name] = df[column_name].rolling(window=window_size).skew()
    features['variance_' + column_name] = df[column_name].rolling(window=window_size).var()
    features['energy_' + column_name] = df[column_name].rolling(window=window_size).apply(lambda x: np.sum(x**2), raw=True)
    features['rms_' + column_name] = df[column_name].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    features['range_' + column_name] = features['max_' + column_name] - features['min_' + column_name]
    return features.dropna()

# Feature extraction for walking and jumping datasets for x, y, and z accelerations
feature_columns = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
features_walk = pd.DataFrame()
features_jump = pd.DataFrame()

for column in feature_columns:
    features_walk = pd.concat([features_walk, calculate_features(walking_df, column)], axis=1)
    features_jump = pd.concat([features_jump, calculate_features(jumping_df, column)], axis=1)

# Drop any rows with NaN values that might occur due to rolling calculations
features_walk = features_walk.dropna().reset_index(drop=True)
features_jump = features_jump.dropna().reset_index(drop=True)

# NORMALIZATION
# Combine walking and jumping features for scaling
all_features = pd.concat([features_walk, features_jump], axis=0)

# Fit the scaler on all data then transform both walking and jumping features
scaler = StandardScaler()
all_features_normalized = scaler.fit_transform(all_features)

# Split the normalized features back into walking and jumping
num_walk = features_walk.shape[0]
normalized_features_walk = all_features_normalized[:num_walk, :]
normalized_features_jump = all_features_normalized[num_walk:, :]

print(normalized_features_walk.shape)
print(normalized_features_jump.shape)

###########################################################################################################

features_walk_labels = np.concatenate((normalized_features_walk, walking_df[['label']].values[:normalized_features_walk.shape[0]]), axis=1)
features_jump_labels = np.concatenate((normalized_features_jump, jumping_df[['label']].values[:normalized_features_jump.shape[0]]), axis=1)

# Combine walking and jumping features and labels for a complete dataset
all_features_labels = np.concatenate((features_walk_labels, features_jump_labels), axis=0)

# Shuffle the combined dataset to ensure a mix of walking and jumping data in both training and testing sets
np.random.shuffle(all_features_labels)

# Split features and labels
X = all_features_labels[:, :-1]  # Features
y = all_features_labels[:, -1]  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy: ', accuracy)
print('Recall: ', recall)