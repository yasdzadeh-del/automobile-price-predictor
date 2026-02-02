# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    # Fixed: Changed _____ to str
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Step 1: Load the model
    # The sweep job outputs an MLflow model, so we point to that folder
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model" 
    # Or more simply, since we are passing the path from the sweep:
    model_uri = args.model_path

    # Step 2 & 3: Log and Register the model
    # This registers the model in the Azure ML Model Registry
    print(f"Registering model from {model_uri}")
    model_details = mlflow.register_model(model_uri, args.model_name)
    
    # Step 4: Write model registration details into a JSON file
    model_info = {
        "model_name": args.model_name,
        "model_version": model_details.version,
        "model_uri": model_details.source
    }
    
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as f:
        json.dump(model_info, f)
    
    print(f"Model info written to {output_path}")

if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    # Fixed: Replaced underscores with actual argument names
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()