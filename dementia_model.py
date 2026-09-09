"""Dementia classification using three standard machine-learning algorithms."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dementia_patients_health_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dementia_model.pkl"

def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns="Dementia"), df["Dementia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1 Score": f1_score(y_test, pred, zero_division=0)})
        fitted[name] = pipe
        print(f"\n{name}\n{classification_report(y_test, pred, target_names=['No Dementia', 'Dementia'], zero_division=0)}")
    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    print("Model Comparison:\n", results_df.to_string(index=False))
    best_name = results_df.iloc[0]["Model"]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(f"\nBest Model: {best_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, fitted[best_name].predict(X_test)))

if __name__ == "__main__":
    main()
