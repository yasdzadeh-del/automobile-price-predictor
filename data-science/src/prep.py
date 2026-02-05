import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to save train data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    return parser.parse_args()

def main(args):
    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Step 1: Handling Categorical Data
    # We use a unique LabelEncoder per column to maintain the mapping
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Step 2: Split
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Step 3: Save (Ensuring directory exists)
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Filenames match train.py: "train.csv" and "test.csv"
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Step 4: Log metrics to the existing Azure ML Run
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))

if __name__ == "__main__":
    args = parse_args()
    main(args)