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
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

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
        # print(window)
        # print('FUCK')
        windows.append(window)
        start_time = end_time
        end_time += window_size

    return windows


############################################################################################################
walking_data_list = pd.concat([oliver_walking_labeled, matthew_walking_labeled, daniel_walking_labeled],
                              ignore_index=True)
walking_data_list = split_data(walking_data_list)

jumping_data_list = pd.concat([oliver_jumping_labeled, matthew_jumping_labeled, daniel_jumping_labeled])
jumping_data_list = split_data(jumping_data_list)

walking_train, walking_test = train_test_split(walking_data_list, test_size=0.1, shuffle=True)

# oliver_walking_split = split_data(oliver_walking_labeled)
# matthew_walking_split = split_data(matthew_walking_labeled)
# daniel_walking_split = split_data(daniel_walking_labeled)

# oliver_jumping_split = split_data(oliver_jumping_labeled)
# matthew_jumping_split = split_data(matthew_jumping_labeled)
# daniel_jumping_split = split_data(daniel_jumping_labeled)

############################################################################################################
# combine dataframes by activity

# walking_df = pd.concat([oliver_walking_split, matthew_walking_split, daniel_walking_split], ignore_index=True)
# jumping_df = pd.concat([oliver_jumping_split, matthew_jumping_split, daniel_jumping_split], ignore_index=True)
############################################################################################################
# Get features of all the data
# Function to calculate features for a specific column

# Recreate an empty DataFrame to store the features


############################################################################################################
# SPLIT data into training and testing sets
# walking_train, walking_test = train_test_split(walking_df, test_size=0.1, shuffle=True)
# jumping_train, jumping_test = train_test_split(jumping_df, test_size=0.1, shuffle=True)

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
    # G41.create_dataset('walking_train', data=walking_train)
    # G41.create_dataset('jumping_train', data=jumping_train)

    G42 = f.create_group('/dataset/test')


# G42.create_dataset('walking_test', data=walking_test)
# G42.create_dataset('jumping_test', data=jumping_test)


############################################################################################################
# Moving average filter
def SMA(data, window_size=5):
    data = data.iloc[:, 1:-1]
    sma = data.rolling(window=window_size).mean()
    sma = sma.dropna()
    return sma


walking_data_list_sma = []
for i in range(len(walking_data_list)):
    walking_data_list_sma.append(SMA(walking_data_list[i], window_size=10))


# jumping_data_list_sma = []
# for i in range(len(jumping_data_list)):
#    jumping_data_list_sma.append(SMA(walking_data_list[i], window_size=5))

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
        features_df = pd.concat([features_df, pd.DataFrame([new_row])], ignore_index=True)

    return features_df  # Corrected indentation for the return statement


def plot_features(features_df1, features_df2):
    num_columns = min(len(features_df1.columns), len(features_df2.columns))
    for i in range(num_columns):
        plt.figure(figsize=(20, 6))

        # Plot regular features
        plt.subplot(1, 2, 1)
        plt.plot(features_df1.iloc[:, i], label=features_df1.columns[i])
        plt.title('Walking - Plot of ' + features_df1.columns[i])
        plt.xlabel('Window')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Plot SMA features
        plt.subplot(1, 2, 2)
        plt.plot(features_df2.iloc[:, i], label=features_df2.columns[i])
        plt.title('Jumping - Plot of ' + features_df2.columns[i])
        plt.xlabel('Window')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.show()


walking_features_df = window_feature_extract(walking_data_list, 'Acceleration z (m/s^2)')
jumping_features_df = window_feature_extract(jumping_data_list, 'Acceleration z (m/s^2)')
walking_features_df['label'] = 0
jumping_features_df['label'] = 1


scaler = StandardScaler()
normalized_walking_features_df = pd.DataFrame(scaler.fit_transform(walking_features_df.drop(columns=['label'])), columns=walking_features_df.drop(columns=['label']).columns)
normalized_jumping_features_df = pd.DataFrame(scaler.transform(jumping_features_df.drop(columns=['label'])), columns=jumping_features_df.drop(columns=['label']).columns)

# Concatenate the normalized DataFrames along the rows
all_features_df = pd.concat([normalized_walking_features_df, normalized_jumping_features_df], ignore_index=True)

# Add labels to the combined DataFrame
all_features_df['label'] = pd.concat([walking_features_df['label'], jumping_features_df['label']], ignore_index=True)

# Shuffle the combined dataset
all_features_df = shuffle(all_features_df)

# Separate features (X) and labels (y)
X = all_features_df.drop(columns=['label'])
y = all_features_df['label']

# Split the data into training and testing sets (90-10 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print accuracy and recall
print('Accuracy:', accuracy)
print('Recall:', recall)
#plot_features(walking_features_df, jumping_features_df)

###########################################################################################################

# features_walk_labels = np.concatenate(
# (normalized_features_walk, walking_df[['label']].values[:normalized_features_walk.shape[0]]), axis=1)
# features_jump_labels = np.concatenate(
#  (normalized_features_jump, jumping_df[['label']].values[:normalized_features_jump.shape[0]]), axis=1)

# Combine walking and jumping features and labels for a complete dataset
# all_features_labels = np.concatenate((features_walk_labels, features_jump_labels), axis=0)

# Shuffle the combined dataset to ensure a mix of walking and jumping data in both training and testing sets
# np.random.shuffle(all_features_labels)

# Split features and labels
# X = all_features_labels[:, :-1]  # Features
# y = all_features_labels[:, -1]  # Labels

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
# model = LogisticRegression(max_iter=10000)
# model.fit(X_train, y_train)

# Predictions
# y_pred = model.predict(X_test)

# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print('Accuracy: ', accuracy)
# print('Recall: ', recall)
