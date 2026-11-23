"""
Alzheimer's Risk Estimator: EDUCATIONAL DEMO ONLY (not a health tool).

Run with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "AlzheimersPredictionModel.joblib"

# ---------------------------------------------------------------------------
# 1. LABEL -> ENCODED VALUE MAPS
#    The dropdowns show the labels on the left; the model receives the numbers
#    on the right. The CSV only contains the numbers, so the label meanings
#    below are ASSUMED (natural ordering: 0 = lowest). If your original
#    preprocessing used a different mapping, fix it HERE and nowhere else.
# ---------------------------------------------------------------------------
GENDER = {"Female": 0, "Male": 1}
LOW_MED_HIGH = {"Low": 0, "Medium": 1, "High": 2}
ALCOHOL = {"Never": 0, "Occasionally": 1, "Regularly": 2}
YES_NO = {"No": 0, "Yes": 1}
CHOLESTEROL = {"Normal": 1, "High": 2}  # data contains only 1 and 2
SLEEP = {"Poor": 0, "Average": 1, "Good": 2}
DIET = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
LIVING = {"Rural": 0, "Urban": 1}

# One-hot groups: the dropdown picks ONE option, which switches on exactly one
# of these columns (and leaves the others at 0).
MARITAL = {
    "Married": "Marital Status_Married",
    "Single": "Marital Status_Single",
    "Widowed": "Marital Status_Widowed",
}
SMOKING = {
    "Never": "Smoking Status_Never",
    "Former": "Smoking Status_Former",
    "Current": "Smoking Status_Current",
}
EMPLOYMENT = {
    "Employed": "Employment Status_Employed",
    "Retired": "Employment Status_Retired",
    "Unemployed": "Employment Status_Unemployed",
}


# ---------------------------------------------------------------------------
# 2. MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_row(values: dict, feature_order) -> pd.DataFrame:
    """Create a one-row DataFrame with columns in the exact order the model was trained on."""
    row = {col: 0 for col in feature_order}
    for col, val in values["direct"].items():
        row[col] = val
    for group, chosen_label in values["onehot"]:
        row[group[chosen_label]] = 1
    return pd.DataFrame([row])[list(feature_order)]


def risk_category(p: float):
    if p < 0.33:
        return "Lower", "success"
    if p < 0.66:
        return "Moderate", "warning"
    return "Higher", "error"


def explain(model, X: pd.DataFrame) -> pd.Series:
    """Per-feature push on the model's log-odds (standardised value x coefficient)."""
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    z = scaler.transform(X)[0]
    return pd.Series(z * clf.coef_[0], index=X.columns).sort_values(
        key=abs, ascending=False
    )


# ---------------------------------------------------------------------------
# 3. PAGE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Alzheimer's Risk Estimator (Demo)", page_icon="🧠")

st.title("🧠 Alzheimer's Risk Estimator")
st.warning(
    "**Educational demo only. This is not a medical tool and not a diagnosis.** "
    "It was built to illustrate a machine learning model on a public dataset. "
    "Its output should not influence any health decision. If you have concerns "
    "about memory or brain health, please speak with a doctor."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Put it next to app.py.")
    st.stop()

model = load_model()

with st.form("inputs"):
    st.subheader("About you")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 50, 94, 70)
        gender = st.selectbox("Gender", list(GENDER))
        education = st.number_input("Years of education", 0, 19, 10)
        bmi = st.number_input("BMI", 18.5, 35.0, 26.0, step=0.1)
        cognitive = st.slider("Cognitive test score", 30, 99, 65)
    with c2:
        marital = st.selectbox("Marital status", list(MARITAL))
        employment = st.selectbox("Employment status", list(EMPLOYMENT))
        income = st.selectbox("Income level", list(LOW_MED_HIGH), index=1)
        living = st.selectbox("Living area", list(LIVING))
        pollution = st.selectbox("Air pollution exposure", list(LOW_MED_HIGH), index=1)

    st.subheader("Health & medical history")
    c3, c4 = st.columns(2)
    with c3:
        diabetes = st.selectbox("Diabetes", list(YES_NO))
        hypertension = st.selectbox("Hypertension", list(YES_NO))
        cholesterol = st.selectbox("Cholesterol level", list(CHOLESTEROL))
        family = st.selectbox("Family history of Alzheimer's", list(YES_NO))
    with c4:
        apoe = st.selectbox("APOE-ε4 gene variant", list(YES_NO))
        depression = st.selectbox("Depression level", list(LOW_MED_HIGH))
        stress = st.selectbox("Stress level", list(LOW_MED_HIGH))

    st.subheader("Lifestyle")
    c5, c6 = st.columns(2)
    with c5:
        activity = st.selectbox("Physical activity level", list(LOW_MED_HIGH), index=1)
        alcohol = st.selectbox("Alcohol consumption", list(ALCOHOL))
        smoking = st.selectbox("Smoking status", list(SMOKING))
    with c6:
        sleep = st.selectbox("Sleep quality", list(SLEEP), index=1)
        diet = st.selectbox("Dietary habits", list(DIET), index=1)
        social = st.selectbox("Social engagement level", list(LOW_MED_HIGH), index=1)

    submitted = st.form_submit_button("Estimate", type="primary")

if submitted:
    values = {
        "direct": {
            "Age": age,
            "Gender": GENDER[gender],
            "Education Level": education,
            "BMI": bmi,
            "Physical Activity Level": LOW_MED_HIGH[activity],
            "Alcohol Consumption": ALCOHOL[alcohol],
            "Diabetes": YES_NO[diabetes],
            "Hypertension": YES_NO[hypertension],
            "Cholesterol Level": CHOLESTEROL[cholesterol],
            "Family History of Alzheimer’s": YES_NO[family],
            "Cognitive Test Score": cognitive,
            "Depression Level": LOW_MED_HIGH[depression],
            "Sleep Quality": SLEEP[sleep],
            "Dietary Habits": DIET[diet],
            "Air Pollution Exposure": LOW_MED_HIGH[pollution],
            "Genetic Risk Factor (APOE-ε4 allele)": YES_NO[apoe],
            "Social Engagement Level": LOW_MED_HIGH[social],
            "Income Level": LOW_MED_HIGH[income],
            "Stress Levels": LOW_MED_HIGH[stress],
            "Urban vs Rural Living": LIVING[living],
        },
        "onehot": [
            (MARITAL, marital),
            (SMOKING, smoking),
            (EMPLOYMENT, employment),
        ],
    }

    X = build_input_row(values, model.feature_names_in_)
    prob = float(model.predict_proba(X)[0, 1])
    label, level = risk_category(prob)

    st.divider()
    st.subheader("Result")
    getattr(st, level)(
        f"**{label} estimated risk** according to this demo model "
        f"(model output: about {prob:.0%})."
    )
    st.progress(prob)
    st.caption(
        "This is a statistical output from a model trained on a public dataset, "
        "not a personal medical prediction. Real risk depends on many things this "
        "model cannot see."
    )

    with st.expander("What influenced this result? (for learning)"):
        contrib = explain(model, X).head(6)
        st.write(
            "Factors pushing the model's estimate the most for these inputs "
            "(positive = pushes the estimate up, negative = pushes it down):"
        )
        st.bar_chart(contrib.rename("influence"))

    with st.expander("Show the exact encoded values sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Educational project. The app does not save or store the information you enter. "
    "The model is a logistic regression trained on a public dataset and has not been "
    "clinically validated. Please do not enter information you are not comfortable sharing."
)
