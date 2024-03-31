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

############################################################################################################
# FUNCTION to label data
def label_data(data_fp, label):
    # read into dataframe
    df = pd.read_csv(data_fp)
    # add label column with label
    df['label'] = label

    return df

# def impute_missing_values(df, strategy='mean'):
#     # create an imputer object with a mean filling strategy
#     imputer = SimpleImputer(strategy=strategy)
#     # fit the imputer object on the data
#     imputer.fit(df)
#     # transform the data
#     imputed_data = imputer.transform(df)
#     # convert back to a data frame
#     imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

#     return imputed_df

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

# # IMPUTE missing values
# oliver_walking_labeled = impute_missing_values(oliver_walking_labeled)
# matthew_walking_labeled = impute_missing_values(matthew_walking_labeled)
# daniel_walking_labeled = impute_missing_values(daniel_walking_labeled)

# oliver_jumping_labeled = impute_missing_values(oliver_jumping_labeled)
# matthew_jumping_labeled = impute_missing_values(matthew_jumping_labeled)
# daniel_jumping_labeled = impute_missing_values(daniel_jumping_labeled)

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
    
    data = data.iloc[:, 1:-1]

    sma = data.rolling(window=window_size).mean()

    sma = sma.dropna()

    return sma

############################################################################################################
# FEATURE EXTRACTION & NORMALIZATION
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
# Concatenate training and testing datasets
train_data = pd.concat([walking_train, jumping_train])
test_data = pd.concat([walking_test, jumping_test])

# Separate features and labels
X_train = train_data.drop('label', axis=1)

# Print how many NaN's in X
print(X_train.isnull().sum())

y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Create a Logistic Regression model
model = LogisticRegression(max_iter=10000)

clf = make_pipeline(StandardScaler(), model)
# Train the model with the imputed training data
clf.fit(X_train, y_train)

# Use the model to make predictions on the imputed test data
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
# Measure the accuracy of your model using the original y_test labels
print('y pred: ', y_pred)
print('y_clf_prob: ', y_clf_prob)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

recall = recall_score(y_test, y_pred)
print('Recall: ', recall)

###########################################################################################################
# PLOT data for walking and jumping, all three axes
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# smoothed_walking = moving_average_filter(walking_df, window_size=5).dropna().reset_index(drop=True)
# smoothed_jumping = moving_average_filter(jumping_df, window_size=5).dropna().reset_index(drop=True)

# # Plotting walking data
# walking_data_sorted = smoothed_walking.sort_values(by='Time (s)')
# walking_data_sorted.plot(x='Time (s)', y='Acceleration x (m/s^2)', ax=axes[0, 0], title='Walking - X acceleration')
# walking_data_sorted.plot(x='Time (s)', y='Acceleration y (m/s^2)', ax=axes[0, 1], title='Walking - Y acceleration')
# walking_data_sorted.plot(x='Time (s)', y='Acceleration z (m/s^2)', ax=axes[0, 2], title='Walking - Z acceleration')

# # Plotting jumping data
# jumping_data_sorted = smoothed_jumping.sort_values(by='Time (s)')
# jumping_data_sorted.plot(x='Time (s)', y='Acceleration x (m/s^2)', ax=axes[1, 0], title='Jumping - X acceleration')
# jumping_data_sorted.plot(x='Time (s)', y='Acceleration y (m/s^2)', ax=axes[1, 1], title='Jumping - Y acceleration')
# jumping_data_sorted.plot(x='Time (s)', y='Acceleration z (m/s^2)', ax=axes[1, 2], title='Jumping - Z acceleration')

# plt.tight_layout()
# plt.show()