import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def process(dataset_folder):
    # Initialize lists to store data
    data = []

    # Read CSV files from the dataset folder
    for file in os.listdir(dataset_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_folder, file), header=None)
            data.append(df)

    # Concatenate all dataframes
    data = pd.concat(data, axis=0, ignore_index=True)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save preprocessed data
    preprocessed_data_folder = "preprocessed_data"
    if not os.path.exists(preprocessed_data_folder):
        os.makedirs(preprocessed_data_folder)

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    X_df.to_csv(os.path.join(preprocessed_data_folder, "X.csv"), index=False)
    y_df.to_csv(os.path.join(preprocessed_data_folder, "y.csv"), index=False)

    print("Preprocessing completed and data saved to preprocessed_data folder.")
