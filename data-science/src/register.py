# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory from sweep output')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    '''Loads the best-trained model and registers it'''

    print(f"Registering model: {args.model_name}")

    # Step 1: Resolve the path
    # We use abspath to ensure MLflow gets the full root-level path on the Azure VM
    model_local_path = os.path.abspath(args.model_path)
    model_uri = f"file://{model_local_path}"

    # Step 2 & 3: Register the model in the Azure ML Registry
    print(f"Registering model from URI: {model_uri}")
    
    # This command creates a new version of the model in your workspace
    model_details = mlflow.register_model(
        model_uri=model_uri, 
        name=args.model_name
    )
    
    # Step 4: Write model registration details into a JSON file for downstream tasks
    model_info = {
        "model_name": args.model_name,
        "model_version": model_details.version,
        "model_uri": model_details.source
    }
    
    # Ensure the output directory exists before writing
    os.makedirs(args.model_info_output_path, exist_ok=True)
    
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Successfully registered version {model_details.version}")
    print(f"Model info written to {output_path}")

if __name__ == "__main__":
    # Start MLflow run to track the registration event
    with mlflow.start_run():
        args = parse_args()
        
        # Displaying inputs for debugging logs
        print(f"Model name: {args.model_name}")
        print(f"Model path: {args.model_path}")
        print(f"Model info output path: {args.model_info_output_path}")

        main(args)