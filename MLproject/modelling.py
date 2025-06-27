import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

if __name__ == "__main__":
    # Ambil parameter dari argumen CLI
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_sample_split = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    data_path = sys.argv[4] if len(sys.argv) > 4 else "Train.csv"

    # Setup URI lokal & experiment name
    # mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    # mlflow.set_experiment("Sub Akhir MSML")

    # Load data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("label", axis=1),
        df["label"],
        test_size=0.2,
        random_state=42
    )
    input_example = X_train[:5]

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_sample_split", min_sample_split)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_sample_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)


