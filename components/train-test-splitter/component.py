import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_data_path: str, training_data_path: str, testing_data_path: str):
    """
    Reads data, drops unneeded columns, splits it, and saves the training and
    testing sets to separate artifact locations.
    """
    df = pd.read_csv(input_data_path)
    if "Id" in df.columns:
        df.drop(columns="Id", inplace=True)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    os.makedirs(training_data_path, exist_ok=True)
    os.makedirs(testing_data_path, exist_ok=True)
    
    X_train.to_csv(f"{training_data_path}/x_train.csv", index=False)
    y_train.to_csv(f"{training_data_path}/y_train.csv", index=False)
    
    X_test.to_csv(f"{testing_data_path}/x_test.csv", index=False)
    y_test.to_csv(f"{testing_data_path}/y_test.csv", index=False)
    
    print(f"Training data saved to {training_data_path}")
    print(f"Testing data saved to {testing_data_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-path', type=str, required=True)
    parser.add_argument('--training-data-path', type=str, required=True)
    parser.add_argument('--testing-data-path', type=str, required=True)
    args = parser.parse_args()
    split_data(args.input_data_path, args.training_data_path, args.testing_data_path)
