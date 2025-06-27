import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
# Create a new MLflow Experiment
mlflow.set_experiment("Sub Akhir MSML")
 
data = pd.read_csv("Credit Card Churn - Processed.csv")
 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("label", axis=1),
    data["label"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

# Mendefinisikan Metode Random Search
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values
 
best_accuracy = 0
best_params = {}
 
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            accuracy = model.score(X_test, y_test)

            # Manual Logging
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)