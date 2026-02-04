# ... (your imports)
import mlflow
import mlflow.sklearn

def main(args):
    # Step 1: Enable autologging 
    # This ensures MLflow captures all scikit-learn metadata automatically
    mlflow.sklearn.autolog()
    
    # Step 2: Read datasets 
    # Reminder: Ensure prep.py saves files exactly as 'train.csv' and 'test.csv'
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # ... (Step 3: Split into X and y)
    target_col = "price" 
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # ... (Step 4: Initialize and train)
    regressor = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    # Step 5: Log metrics (Matches 'MSE' in your newpipeline.yml)
    predictions = regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("MSE", mse)
    print(f"MSE: {mse}")

    # Step 6: Save the model
    if os.path.exists(args.model_output):
        shutil.rmtree(args.model_output)
    
    # This saves the model in the format the 'register_model' step expects
    mlflow.sklearn.save_model(sk_model=regressor, path=args.model_output)