import argparse
from pathlib import Path
import mlflow
import os 
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory from sweep output')
    parser.add_argument("--model_info_output_path", type=str, default=None, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    print(f"Registering model: {args.model_name}")
    
    # Force Python to handle the path as a normalized string
    base_search_path = os.path.normpath(args.model_path)
    print(f"Target search path: {base_search_path}")

    model_abs_path = None

    # Step 1: Broad Search
    # We check the provided path, the parent, and the current working directory
    search_locations = [base_search_path, os.path.dirname(base_search_path), os.getcwd()]
    
    for location in search_locations:
        if os.path.exists(location):
            print(f"Checking location: {location}")
            for root, dirs, files in os.walk(location):
                if "MLmodel" in files:
                    model_abs_path = root
                    print(f"Found MLmodel at: {model_abs_path}")
                    break
        if model_abs_path:
            break

    # Step 2: The "Hail Mary" Search
    # If still not found, search the entire Azure ML mount drive
    if not model_abs_path:
        print("!!! Model not found in primary paths. Scanning /mnt/azureml/ !!!")
        for root, dirs, files in os.walk("/mnt/azureml/"):
            if "MLmodel" in files:
                model_abs_path = root
                print(f"FOUND IT AT MOUNT PATH: {model_abs_path}")
                break
    
    # Final Validation
    if not model_abs_path:
        # Final debug listing to see what the container actually sees
        print("Directory content listing of base path:")
        for root, dirs, files in os.walk(base_search_path):
             print(f"Path: {root} | Files: {files}")
        raise FileNotFoundError(f"Could not find MLmodel file anywhere in /mnt/azureml/ for path {args.model_path}")

    model_uri = f"file://{model_abs_path}"
    
    # Step 3: Register the model
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