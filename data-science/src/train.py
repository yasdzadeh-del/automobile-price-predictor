# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from pathlib import Path
import pandas as pd
import shutil
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path to save the model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees")

    args = parser.parse_args()
    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    
    # Step 2: Read datasets 
    # Using Path for cross-platform compatibility
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Step 3: Split into features (X) and target (y)
    # Ensure 'price' matches your dataset column name exactly
    target_col = "price" 
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Step 4: Initialize and train RandomForest
    regressor = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    # Step 5: Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Predict and calculate MSE
    predictions = regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("MSE", mse)
    print(f"MSE: {mse}")

    # Step 7: Save the trained model
    # MLflow save_model fails if the directory exists. 
    # We check and remove it if it's there to allow MLflow to write.
    if os.path.exists(args.model_output):
        shutil.rmtree(args.model_output)
    
    mlflow.sklearn.save_model(sk_model=regressor, path=args.model_output)
    print(f"Model successfully saved to {args.model_output}")

if __name__ == "__main__":
    # In Azure ML Pipelines, a run is often already active. 
    # This check prevents nested run errors.
    if mlflow.active_run() is None:
        mlflow.start_run()

    args = parse_args()

    # Debugging logs for the Azure Job log stream
    print(f"Received Train path: {args.train_data}")
    print(f"Received Model output path: {args.model_output}")

    main(args)

    # Only end the run if we started it here
    if mlflow.active_run():
        mlflow.end_run()