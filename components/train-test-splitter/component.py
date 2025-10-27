import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_dataset_path: str, training_data_path: str, testing_data_path: str):
    input_file = os.path.join(input_dataset_path, "data.csv")
    print(f"Loading data from {input_file} to split.")
    df = pd.read_csv(input_file)
    if "Id" in df.columns:
        print("Found and removed the 'Id' column.")
        df.drop(columns="Id", inplace=True)
    print("Separating features (X) from the target (y).")
    X = df.drop(columns=["quality"])
    y = df["quality"]
    print("Performing an 80/20 train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(training_data_path, exist_ok=True)
    os.makedirs(testing_data_path, exist_ok=True)
    print("Saving split datasets to output paths.")
    X_train.to_csv(os.path.join(training_data_path, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(training_data_path, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(testing_data_path, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(testing_data_path, "y_test.csv"), index=False)
    print("Data splitting and saving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-path", type=str, required=True)
    parser.add_argument("--training-data-path", type=str, required=True)
    parser.add_argument("--testing-data-path", type=str, required=True)
    args = parser.parse_args()
    split_data(args.input_dataset_path, args.training_data_path, args.testing_data_path)