import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # 1. Chargement
    df = pd.read_csv(filepath)

    # 2. Suppression de l'identifiant inutile
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 3. Nettoyage de TotalCharges (contient des espaces vides " ")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 4. Encodage de la variable cible (Yes -> 1, No -> 0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. Séparation features / target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 6. Encodage One-Hot des variables catégorielles
    X = pd.get_dummies(X, drop_first=True)

    # 7. Triple découpage stratifié : 70% Train, 15% Validation, 15% Test
    # (random_state=42 fixe pour garantir l'identité entre les deux binômes)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 8. Normalisation (fit uniquement sur Train pour éviter le data leakage)
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test, y_train, y_val, y_test