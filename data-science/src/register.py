import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory from sweep output')
    # Made this optional to prevent errors if not defined in YAML
    parser.add_argument("--model_info_output_path", type=str, default=None, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    print(f"Registering model: {args.model_name}")

    # Step 1: Use the path directly
    # MLflow register_model accepts a local path string without the file:// prefix
    model_local_path = os.path.abspath(args.model_path)

    # Step 2 & 3: Register the model
    print(f"Registering model from path: {model_local_path}")
    
    model_details = mlflow.register_model(
        model_uri=model_local_path, 
        name=args.model_name
    )
    
    # Step 4: Write model info ONLY if a path was provided in the pipeline
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
    # Using 'with' handles the mlflow.end_run() automatically even if code fails
    with mlflow.start_run():
        args = parse_args()
        main(args)