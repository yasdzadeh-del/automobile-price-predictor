# ... (rest of imports)

def main(args):
    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Step 1: Label Encoding
    # Note: LabelEncoder is great for target variables, 
    # but for features, consider that it doesn't handle 'new' categories 
    # in future inference well. For now, this is fine for training.
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # ... (Step 2 & 3: Split and Save)
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # These filenames match your train.py requirements perfectly
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Step 4: Log metrics
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))

if __name__ == "__main__":
    # Check for active run to prevent errors in nested pipeline steps
    if mlflow.active_run() is None:
        mlflow.start_run()

    args = parse_args()
    main(args)

    if mlflow.active_run():
        mlflow.end_run()