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

    print(f"Registering model: {args.model_name}")

    # Step 1: Point to the model path provided by the sweep output
    # Since the sweep output is an MLflow model, we can use the local path directly
    # with the 'file://' prefix or just the path string in many environments.
    model_path = args.model_path

    # Step 2 & 3: Register the model
    # We use mlflow.log_artifact to ensure the model is associated with THIS run too,
    # then register it. Or, register directly from the path:
    print(f"Registering model from path: {model_path}")
    
    # Registering the model using the path provided by the pipeline binding
    model_details = mlflow.register_model(
        model_uri=f"file://{os.path.abspath(model_path)}", 
        name=args.model_name
    )
    
    # Step 4: Write model registration details into a JSON file
    model_info = {
        "model_name": args.model_name,
        "model_version": model_details.version,
        "model_uri": model_details.source
    }
    
    # Ensure the output directory exists
    os.makedirs(args.model_info_output_path, exist_ok=True)
    
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as f:
        json.dump(model_info, f)
    
    print(f"Model info written to {output_path}")