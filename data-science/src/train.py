import os
import argparse
import pandas as pd
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--model_output", type=str, help="Path to save model")
    return parser.parse_args()

def main(args):
    # Step 1: Enable autologging 
    mlflow.sklearn.autolog()
    
    # Step 2: Read datasets 
    # Logic: Assumes prep.py used os.path.join(args.train_data, "train.csv")
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Step 3: Split into X and y
    # Note: Using 'price' to match your previous context
    target_col = "price" 
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Step 4: Initialize and train
    regressor = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    # Step 5: Log metrics (Matches 'MSE' primary_metric in newpipeline.yml)
    predictions = regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("MSE", mse)
    print(f"MSE: {mse}")

    # Step 6: Save the model
    os.makedirs(args.model_output, exist_ok=True)
    
    # We save directly to the output path provided by the sweep trial
    mlflow.sklearn.save_model(sk_model=regressor, path=args.model_output)
    print(f"Model saved to {args.model_output}")