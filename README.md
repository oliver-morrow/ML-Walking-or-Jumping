# ELEC 292 Final Project
Welcome to the final project created by Oliver, Matthew, and Daniel.
# DATA_ANALYSIS
This folder contains all of the files and scripts needed to complete steps 1-5 of the data analysis portion of the project. This includes:
- **split.py**: Split the walking file into 5 second intervals, split the jumping file into 5 second intervals.
- **concat.py**: Concatenate the already split walking file with the already split jumping file.
The **main.py** file takes in one CSV file per person which contains their split and concatenated data, then concatenates them all together, then shuffles the data. It then performs pre-processing, visualization, feature extraction, normalization. It then puts the CSV files into an HDF5 file format with the format shown:
<img width="500" alt="image" src="https://github.com/oliver-morrow/ELEC292-Final-Project/assets/83565270/efe43c65-da52-492d-806e-821989bf0237">

It then splits the model data by 90% training, 10% testing using scikit-learn modules.
# DESKTOP_APP
This folder contains all files and scripts needed to complete steps 6-7 of the desktop app portion of the project. This includes:
- **main.py**: Loads the model data as a .joblib file, and accepts a CSV from the user and using the model, it predicts walking or jumping as a new column in the CSV and returns the modified CSV to the user.
