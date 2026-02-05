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

    # Step 1: Search for the MLmodel file recursively
    # This bypasses the broken ${{name}} path by searching for the file itself
    search_pattern = os.path.join(args.model_path, "**", "MLmodel")
    mlmodel_files = glob.glob(search_pattern, recursive=True)

    if not mlmodel_files:
        # Debugging: if it fails, this will show us what Azure actually sent
        print(f"Current directory structure of {args.model_path}:")
        for root, dirs, files in os.walk(args.model_path):
            print(f"  {root}: {files}")
        raise FileNotFoundError(f"Could not find MLmodel file in {args.model_path}")

    # Use the first MLmodel file found (the best trial output)
    model_abs_path = os.path.dirname(mlmodel_files[0])
    model_uri = f"file://{model_abs_path}"

    print(f"Registering model from URI: {model_uri}")
    
    # Step 2 & 3: Register the model
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