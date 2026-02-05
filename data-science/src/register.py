import argparse
from pathlib import Path
import mlflow
import os 
import json
import glob  # Moved to the top

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory from sweep output')
    parser.add_argument("--model_info_output_path", type=str, default=None, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    print(f"Registering model: {args.model_name}")
    print(f"Searching for MLmodel in: {args.model_path}")

    model_abs_path = None

    # Step 1: Manually walk through the directory to find MLmodel
    for root, dirs, files in os.walk(args.model_path):
        if "MLmodel" in files:
            model_abs_path = root
            print(f"Found MLmodel at: {model_abs_path}")
            break

    if not model_abs_path:
        # This will help us see EXACTLY what is in there if it fails
        print("Directory content listing:")
        for root, dirs, files in os.walk(args.model_path):
            print(f"Path: {root} | Files: {files}")
        raise FileNotFoundError(f"Could not find MLmodel file in {args.model_path}")

    model_uri = f"file://{model_abs_path}"
    
    # Step 2 & 3: Register the model
    print(f"Registering model from URI: {model_uri}")
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
    with mlflow.start_run():
        args = parse_args()
        main(args)