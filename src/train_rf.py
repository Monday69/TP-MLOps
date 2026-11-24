import mlflow
from preprocessing import load_and_preprocess_data

# Configuration MLflow imposée par le sujet
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Customer_Churn_Optuna")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    print(f"Données chargées : Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print("Connexion MLflow OK !")