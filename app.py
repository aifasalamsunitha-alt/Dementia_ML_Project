from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dementia_patients_health_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dementia_model.pkl"

st.set_page_config(page_title="Dementia Classification", page_icon="🧠")
st.title("🧠 Dementia Classification")
st.caption("Educational machine-learning demonstration — not a medical diagnosis")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Dataset or trained model is missing. Run: python dementia_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != "Dementia"]

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
    input_df = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_df)[0])
    if prediction == 1:
        st.error("Prediction: Dementia")
    else:
        st.success("Prediction: No Dementia")
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
        st.write(f"Model probability for Dementia: **{probability:.2%}**")
