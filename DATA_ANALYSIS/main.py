import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from joblib import dump

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
        
    return windows

############################################################################################################
walking_data_list = pd.concat([oliver_walking_labeled, matthew_walking_labeled])
walking_data_list = split_data(walking_data_list)

jumping_data_list = pd.concat([oliver_jumping_labeled, matthew_jumping_labeled])
jumping_data_list = split_data(jumping_data_list)


# Append lists
data_list = walking_data_list + jumping_data_list
train, test = train_test_split(data_list, test_size=0.1, shuffle=True)

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
    
    for i in range(len(train)):
        dataset = G41.create_dataset(f'window_{i}', data=train[i])

    G42 = f.create_group('/dataset/test')
    for i in range(len(test)):
        dataset = G42.create_dataset(f'window_{i}', data=test[i])

    f.close()

with h5py.File('data.h5', 'r') as f:
    # Access the data
    oliver_walking_data = f['/oliver/walking'][()]
    oliver_jumping_data = f['/oliver/jumping'][()]
    matthew_walking_data = f['/matthew/walking'][()]
    matthew_jumping_data = f['/matthew/jumping'][()]
    daniel_walking_data = f['/daniel/walking'][()]
    daniel_jumping_data = f['/daniel/jumping'][()]

############################################################################################################
# Moving average filter
def SMA(data, columns, window_size):
    smoothed_data = data.copy()
    for col in columns:
        # Use rolling window to calculate the moving average, then fill NaNs at the start with original values
        smoothed_col = data[col].rolling(window=window_size, min_periods=1).mean()
        smoothed_data[col] = smoothed_col

    return smoothed_data

sensor_columns = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']

# Apply the moving average to each window in walking and jumping data lists
walking_data_list_sma = [SMA(window, sensor_columns, window_size=5) for window in walking_data_list]
jumping_data_list_sma = [SMA(window, sensor_columns, window_size=5) for window in jumping_data_list]

# def plot_acceleration_data(window, window_index):
#     plt.figure(figsize=(14, 6))
    
#     # Ensure the window is sorted by 'Time (s)' to avoid crisscross lines
#     window = window.sort_values(by='Time (s)')
    
#     time = window['Time (s)']
#     acc_x = window['Acceleration x (m/s^2)']
#     acc_y = window['Acceleration y (m/s^2)']
#     acc_z = window['Acceleration z (m/s^2)']
    
#     plt.plot(time, acc_x, label='Acceleration x (m/s^2)')
#     plt.plot(time, acc_y, label='Acceleration y (m/s^2)')
#     plt.plot(time, acc_z, label='Acceleration z (m/s^2)')
    
#     plt.title(f'Window {window_index}: Acceleration Data')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Acceleration (m/s^2)')
#     plt.legend()
#     plt.show()

# # Plot the first three windows from the walking data list
# for i, window in enumerate(walking_data_list_sma[:3]):
#     plot_acceleration_data(window, i+1)

# def plot_all_windows_combined(windows, sensor_columns):
#     plt.figure(figsize=(18, 12))
#     colors = ['r', 'g', 'b']  # Colors for x, y, z axes
#     labels = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
    
#     # Create a plot for each axis
#     for i, col in enumerate(sensor_columns):
#         plt.subplot(3, 1, i+1)
#         for window in windows:
#             if 'Time (s)' in window.columns and col in window.columns:
#                 # Ensure the window is sorted by 'Time (s)'
#                 window_sorted = window.sort_values(by='Time (s)')
#                 plt.plot(window_sorted['Time (s)'], window_sorted[col], color=colors[i], alpha=0.5)
                
#         plt.title(labels[i])
#         plt.xlabel('Time (s)')
#         plt.ylabel('Acceleration (m/s^2)')
#         plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# # Assuming 'walking_data_list_sma' contains your smoothed windows
# plot_all_windows_combined(walking_data_list, ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'])
############################################################################################################

def window_feature_extract(window_list, columns):
    features_df = pd.DataFrame()

    for column in columns:
        column_features_df = pd.DataFrame(columns=['max_' + column, 'min_' + column, 'skew_' + column, 'std_' + column, \
                                                'mean_' + column, 'median_' + column, 'variance_' + column, 'range_' + column, \
                                                'energy_' + column, 'rms_' + column, ])

        # Iterate over each window in window_list and calculate features
        for i in range(len(window_list)):
            f_max = window_list[i][column].max()
            f_min = window_list[i][column].min()
            skew = window_list[i][column].skew()
            std = window_list[i][column].std()
            mean = window_list[i][column].mean()
            median = window_list[i][column].median()
            variance = window_list[i][column].var()
            f_range = f_max - f_min
            energy = np.sum(window_list[i][column] ** 2)
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

        features_df = pd.concat([features_df, column_features_df], axis=1)

    return features_df

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


columns = ['Absolute acceleration (m/s^2)', 'Acceleration z (m/s^2)']
walking_features_df = window_feature_extract(walking_data_list_sma, columns)
jumping_features_df = window_feature_extract(jumping_data_list_sma, columns)
walking_features_df['label'] = 'Walking'
jumping_features_df['label'] = 'Jumping'

# scaler = StandardScaler()
# normalized_walking_features_df = pd.DataFrame(scaler.fit_transform(walking_features_df.drop(columns=['label'])), columns=walking_features_df.drop(columns=['label']).columns)
# normalized_jumping_features_df = pd.DataFrame(scaler.transform(jumping_features_df.drop(columns=['label'])), columns=jumping_features_df.drop(columns=['label']).columns)

# Concatenate the normalized DataFrames along the rows
all_features_df = pd.concat([walking_features_df, jumping_features_df], ignore_index=True)
# Add labels to the combined DataFrame
all_features_df['label'] = pd.concat([walking_features_df['label'], jumping_features_df['label']], ignore_index=True)
# Shuffle the combined dataset
all_features_df = shuffle(all_features_df)

# Separate features (X) and labels (y)
data = all_features_df.drop(columns=['label'])
labels = all_features_df['label']

# Split the data into training and testing sets (90-10 split)
X_train, X_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.1, random_state=0)

scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='Jumping')

# Print accuracy and recall
print('Accuracy:', accuracy)
print('Recall:', recall)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Plotting the ROC curve
y_clf_prob = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# calculating AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print('AUC:', auc)

# Save the model to a file
from joblib import dump
dump(clf, 'model.joblib')

plot_features(walking_features_df, jumping_features_df)

pca = PCA(n_components=2)
pca_pipe = make_pipeline(StandardScaler(), pca)
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.fit_transform(X_test)

clf = make_pipeline(LogisticRegression(max_iter=10000))
clf.fit(X_train_pca, y_train)

disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method='predict', xlabel='X1', ylabel='X2', alpha=0.5,
)

label_mapping = {'Walking': 0, 'Jumping': 1}
y_train_numeric = y_train.map(label_mapping)

disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_numeric)
plt.show()
