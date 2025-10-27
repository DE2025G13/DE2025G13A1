import argparse
import os
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split

def split_data(github_repo_url: str, github_branch: str, output_dataset_path: str):
    # Clone the repository
    repo_dir = "/tmp/repo"
    print(f"Cloning repository: {github_repo_url}, branch: {github_branch}")
    subprocess.run([
        "git", "clone", 
        "--branch", github_branch,
        "--depth", "1",  # Only get the latest commit for speed
        github_repo_url,
        repo_dir
    ], check=True)
    input_file = os.path.join(repo_dir, "wine.csv")
    print(f"Loading data from {input_file}")
    # The new dataset uses semicolons as a separator
    df = pd.read_csv(input_file, sep=";")
    # Check for the 'Id' column and drop it if it exists
    if "Id" in df.columns:
        print("Found and removed the 'Id' column.")
        df.drop(columns="Id", inplace=True)
    print("Encoding the 'type' column (white=0, red=1).")
    df["type"] = df["type"].apply(lambda x: 1 if x == "red" else 0)
    print("Separating features (X) from the target (y).")
    X = df.drop(columns=["quality"])
    y = df["quality"]
    print("Performing an 80/20 train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(output_dataset_path, exist_ok=True)
    print("Saving split datasets to output paths.")
    X_train.to_csv(os.path.join(output_dataset_path, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dataset_path, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dataset_path, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dataset_path, "y_test.csv"), index=False)
    print("Data splitting and saving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-repo-url", type=str, required=True)
    parser.add_argument("--github-branch", type=str, default="dataset")
    parser.add_argument("--output-dataset-path", type=str, required=True)
    args = parser.parse_args()
    split_data(
        args.github_repo_url,
        args.github_branch,
        args.output_dataset_path
    )