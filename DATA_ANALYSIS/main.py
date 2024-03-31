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

def impute_missing_values(df, strategy='mean'):
    # create an imputer object with a mean filling strategy
    imputer = SimpleImputer(strategy=strategy)
    # fit the imputer object on the data
    imputer.fit(df)
    # transform the data
    imputed_data = imputer.transform(df)
    # convert back to a data frame
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

    return imputed_df

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

# IMPUTE missing values
oliver_walking_labeled = impute_missing_values(oliver_walking_labeled)
matthew_walking_labeled = impute_missing_values(matthew_walking_labeled)
daniel_walking_labeled = impute_missing_values(daniel_walking_labeled)

oliver_jumping_labeled = impute_missing_values(oliver_jumping_labeled)
matthew_jumping_labeled = impute_missing_values(matthew_jumping_labeled)
daniel_jumping_labeled = impute_missing_values(daniel_jumping_labeled)

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
features_walk = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'median', 'skew', 'variance', 'energy', 'rms', 'range'])

# Calculate rolling features for 'Absolute acceleration (m/s^2)'
features_walk['mean'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).mean()
features_walk['std'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).std()
features_walk['max'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).max()
features_walk['min'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).min()
features_walk['median'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).median()
features_walk['skew'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).skew()
# features['kurtosis'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).kurtosis()
features_walk['variance'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).var().dropna()
features_walk['energy'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sum(x**2), raw=True)
features_walk['rms'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

# Calculate the range as max - min
features_walk['range'] = features_walk['max'] - features_walk['min']

# Drop NaN values from the DataFrame
newfeatures_walk = features_walk.dropna().reset_index(drop=True)

# Now print the features DataFrame
print(newfeatures_walk)


features_jump = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'median', 'skew', 'variance', 'energy', 'rms', 'range'])

# Calculate rolling features for 'Absolute acceleration (m/s^2)'
features_jump['mean'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).mean()
features_jump['std'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).std()
features_jump['max'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).max()
features_jump['min'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).min()
features_jump['median'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).median()
features_jump['skew'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).skew()
# features['kurtosis'] = walking_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).kurtosis()
features_jump['variance'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).var().dropna()
features_jump['energy'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sum(x**2), raw=True)
features_jump['rms'] = jumping_df['Absolute acceleration (m/s^2)'].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
features_jump['range'] = features_jump['max'] - features_jump['min']

# Drop NaN values from the DataFrame
newfeatures_jump = features_jump.dropna().reset_index(drop=True)
# Now print the features DataFrame
print(newfeatures_jump)

# NORMALIZATION
scaler = StandardScaler()
normalized_features_walk = scaler.fit_transform(newfeatures_walk)
normalized_features_jump = scaler.fit_transform(newfeatures_jump)
print(normalized_features_walk)
print(normalized_features_jump)
###########################################################################################################
# Concatenate training and testing datasets
train_data = pd.concat([walking_train, jumping_train])
test_data = pd.concat([walking_test, jumping_test])

# Separate features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Create a Logistic Regression model
model = LogisticRegression(max_iter=10000)

clf = make_pipeline(StandardScaler(), model)
# Train the model with the imputed training data
model.fit(X_train, y_train)

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