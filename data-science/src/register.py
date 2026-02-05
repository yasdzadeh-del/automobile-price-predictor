import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory from sweep output')
    parser.add_argument("--model_info_output_path", type=str, default=None, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    print(f"Registering model: {args.model_name}")

    # Step 1: Format the path for MLflow
    # Ensure it's an absolute path and add the 'file://' prefix required for local folder registration
    model_abs_path = os.path.abspath(args.model_path)
    
    # Validation: Check if the MLmodel file exists in the folder
    if not os.path.exists(os.path.join(model_abs_path, "MLmodel")):
        raise FileNotFoundError(f"No MLmodel file found at {model_abs_path}. Check if train.py saved it correctly.")

    model_uri = f"file://{model_abs_path}"

    # Step 2 & 3: Register the model
    print(f"Registering model from URI: {model_uri}")
    
    # MLflow in Azure ML automatically uses the workspace's registry
    model_details = mlflow.register_model(
        model_uri=model_uri, 
        name=args.model_name
    )
    
    # Step 4: Write model info ONLY if a path was provided
    if args.model_info_output_path:
        model_info = {
            "model_name": args.model_name,
            "model_version": model_details.version,
            "model_uri": model_details.source
        }
        
        os.makedirs(args.model_info_output_path, exist_ok=True)
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=4)
        print(f"Model info written to {output_path}")
    
    print(f"Successfully registered version {model_details.version}")

if __name__ == "__main__":
    # We use the active run context to link the registration to the pipeline run
    with mlflow.start_run():
        args = parse_args()
        main(args)