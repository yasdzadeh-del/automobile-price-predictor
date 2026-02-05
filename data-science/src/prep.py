import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def main(args):
    print("--- STARTING PREP ---")
    print(f"Checking for data at: {args.raw_data}")
    
    if not os.path.exists(args.raw_data):
        print(f"ERROR: Raw data file not found at {args.raw_data}")
        # List files in the parent directory to see what Azure mounted
        parent = os.path.dirname(args.raw_data)
        print(f"Files in {parent}: {os.listdir(parent) if os.path.exists(parent) else 'Dir not found'}")
    
    df = pd.read_csv(args.raw_data)
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns found: {df.columns.tolist()}")

    # Force columns to lowercase to prevent 'Price' vs 'price' errors
    df.columns = [c.lower() for c in df.columns]

    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)
    
    print("--- PREP COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)