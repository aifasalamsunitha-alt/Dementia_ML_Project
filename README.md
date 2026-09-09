# Classification and Analysis of Dementia Using Machine Learning Algorithms

An educational machine-learning project that classifies the `Dementia` target from patient health data.

## 🚀 Live Streamlit App

**Try the deployed application:**

👉 https://dementiamlproject-6t69qh5sgnadxlldbofpof.streamlit.app/

## Dataset
- 1,000 records and 24 columns
- Target: `Dementia` (0 = No Dementia, 1 = Dementia)
- Numerical and categorical health-related features
- Missing values are handled during preprocessing

## Machine Learning Workflow
1. Load and inspect the dataset
2. Remove duplicate rows
3. Separate features and target
4. Impute missing values
5. Scale numerical features
6. One-hot encode categorical features
7. Split data into 80% training and 20% testing
8. Train Logistic Regression, Decision Tree, and Random Forest
9. Compare Accuracy, Precision, Recall, and F1-score
10. Save the best complete pipeline as `models/dementia_model.pkl`

## Results on the supplied dataset
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 0.995 | 1.000 | 0.990 | 0.995 |

Results use an 80/20 stratified split with `random_state=42`. The unusually high scores should not be interpreted as proof of clinical performance or real-world generalization.

## Project Structure
```text
Dementia_ML_Project/
├── dementia_patients_health_data.csv
├── dementia_model.py
├── app.py
├── requirements.txt
├── README.md
└── models/
    └── dementia_model.pkl
```

## Run the project
```bash
pip install -r requirements.txt
python dementia_model.py
streamlit run app.py
```

## Disclaimer
This project is for academic and educational demonstration only. It must not be used to diagnose dementia or make medical decisions.
