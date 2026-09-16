
   from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# scikit-learn imports for on-the-fly model fallback
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dementia_patients_health_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dementia_model.pkl"

st.set_page_config(page_title="Dementia Classification", page_icon="🧠")
st.title("🧠 Dementia Classification")
st.caption("Educational machine-learning demonstration — not a medical diagnosis")

def build_and_save_model(df_path, model_path):
    """Trains a fresh model matched to the current runtime environment."""
    df = pd.read_csv(df_path).drop_duplicates().reset_index(drop=True)
    X = df.drop(columns="Dementia")
    y = df["Dementia"]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

    model.fit(X, y)
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    return model

@st.cache_resource
def load_or_train_model():
    if not DATA_PATH.exists():
        st.error("Dataset missing. Ensure `dementia_patients_health_data.csv` is present.")
        st.stop()

    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # Re-train if unpickling fails due to version mismatch
            return build_and_save_model(DATA_PATH, MODEL_PATH)
    else:
        return build_and_save_model(DATA_PATH, MODEL_PATH)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

model = load_or_train_model()
df = load_data(DATA_PATH)
features = [c for c in df.columns if c.strip().lower() != "dementia"]

st.write("Enter patient information and click **Predict**.")

with st.form("prediction_form"):
    values = {}
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            default = float(s.median()) if s.notna().any() else 0.0
            if pd.api.types.is_integer_dtype(s.dropna()):
                values[col] = st.number_input(col, value=int(default))
            else:
                values[col] = st.number_input(col, value=default, format="%.2f")
        else:
            options = sorted(s.dropna().astype(str).unique().tolist()) or [""]
            values[col] = st.selectbox(col, options)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(input_df)[0])

        if prediction == 1:
            st.error("Prediction: Dementia")
        else:
            st.success("Prediction: No Dementia")

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])
            st.write(f"Model probability for Dementia: **{probability:.2%}**")

    except Exception as e:
        st.error("An error occurred during inference. Ensure feature formatting matches model expectations.")
        st.exception(e)
