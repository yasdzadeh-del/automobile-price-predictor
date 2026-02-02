# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''
    # Fixed: ArgumentParser initialization
    parser = argparse.ArgumentParser("prep")  
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    
    args = parser.parse_args()
    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Step 1: Label Encoding for categorical features
    # This converts columns like 'make' or 'fuel_type' into numbers
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Step 2: Split the dataset
    train_df, test_df = train_test_split(
        df, 
        test_size=args.test_train_ratio, 
        random_state=42
    )

    # Step 3: Save datasets as CSV files
    # We create the directories first to avoid 'Folder not found' errors
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Step 4: Log metrics to MLflow
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))
    
    print(f"Data prepared: {len(train_df)} train rows, {len(test_df)} test rows.")

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()