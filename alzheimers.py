"""
Alzheimer's ML Project: five-page educational web app (NOT a health tool).
Run with:  streamlit run alzheimers.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "AlzheimersPredictionModel.joblib"

# ===========================================================================
# 1. LABEL -> ENCODED VALUE MAPS
#    The dropdowns show the labels; the model receives the numbers. The CSV
#    contains only numbers, so these label meanings are ASSUMED (natural order,
#    0 = lowest). If your preprocessing differed, fix the mapping HERE only.
# ===========================================================================
GENDER = {"Female": 0, "Male": 1}
LMH = {"Low": 0, "Medium": 1, "High": 2}
ALCOHOL = {"Never": 0, "Occasionally": 1, "Regularly": 2}
YES_NO = {"No": 0, "Yes": 1}
CHOLESTEROL = {"Normal": 1, "High": 2}  # the data contains only 1 and 2
SLEEP = {"Poor": 0, "Average": 1, "Good": 2}
DIET = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
LIVING = {"Rural": 0, "Urban": 1}
# One-hot groups: one dropdown -> exactly one of these model columns is set to 1
MARITAL = {"Married": "Marital Status_Married", "Single": "Marital Status_Single",
           "Widowed": "Marital Status_Widowed"}
SMOKING = {"Never": "Smoking Status_Never", "Former": "Smoking Status_Former",
           "Current": "Smoking Status_Current"}
EMPLOYMENT = {"Employed": "Employment Status_Employed", "Retired": "Employment Status_Retired",
              "Unemployed": "Employment Status_Unemployed"}

# Form layout. Each field: (key, label, model feature or None for one-hot, spec, help)
# spec = dict -> dropdown;  spec = (min, max, step) -> number box.
# Number limits are the min/max found in the full training dataset.
SECTIONS = [
    ("A. Personal information", [
        ("age", "Age (years)", "Age", (50, 94, 1), "The training data covers ages 50 to 94."),
        ("gender", "Gender", "Gender", GENDER, "As recorded in the dataset (female or male)."),
        ("education", "Education (years)", "Education Level", (0, 19, 1),
         "Years of formal education, from 0 to 19 in the training data."),
        ("bmi", "BMI", "BMI", (18.5, 35.0, 0.1),
         "Body mass index: weight in kg divided by height in metres squared. Training range 18.5 to 35."),
        ("marital", "Marital status", None, MARITAL, ""),
        ("employment", "Employment status", None, EMPLOYMENT, ""),
    ]),
    ("B. Medical history", [
        ("diabetes", "Diabetes", "Diabetes", YES_NO, ""),
        ("hypertension", "Hypertension (high blood pressure)", "Hypertension", YES_NO, ""),
        ("cholesterol", "Cholesterol level", "Cholesterol Level", CHOLESTEROL,
         "Whether cholesterol is in the normal range or high."),
        ("family", "Family history of Alzheimer's", "Family History of Alzheimer’s", YES_NO,
         "Yes if a parent or sibling has had Alzheimer's disease."),
        ("apoe", "Genetic risk factor (APOE-ε4)", "Genetic Risk Factor (APOE-ε4 allele)", YES_NO,
         "APOE-ε4 is a version of the APOE gene that is more common in people with Alzheimer's. "
         "Answer Yes only if a genetic test showed you carry it; otherwise choose No."),
    ]),
    ("C. Lifestyle", [
        ("activity", "Physical activity level", "Physical Activity Level", LMH, ""),
        ("alcohol", "Alcohol consumption", "Alcohol Consumption", ALCOHOL, ""),
        ("smoking", "Smoking status", None, SMOKING, ""),
        ("sleep", "Sleep quality", "Sleep Quality", SLEEP, ""),
        ("diet", "Dietary habits", "Dietary Habits", DIET, ""),
        ("social", "Social engagement level", "Social Engagement Level", LMH,
         "How often the person takes part in social activities."),
        ("stress", "Stress level", "Stress Levels", LMH, ""),
    ]),
    ("D. Environmental and cognitive factors", [
        ("cognitive", "Cognitive test score", "Cognitive Test Score", (30, 99, 1),
         "A numeric score from a cognitive test, on the scale used in the dataset (30 to 99)."),
        ("depression", "Depression level", "Depression Level", LMH, ""),
        ("pollution", "Air pollution exposure", "Air Pollution Exposure", LMH, ""),
        ("income", "Income level", "Income Level", LMH, ""),
        ("living", "Urban or rural living", "Urban vs Rural Living", LIVING, ""),
    ]),
]
FIELDS = [f for _, fields in SECTIONS for f in fields]

# Evaluation figures. Reproduced with the saved model on a stratified 70/30
# train/test split (random_state=42), positive class = 1. This assumes the model
# was trained on the 70% partition.
METRICS = {"Accuracy": 0.7087, "Precision": 0.6637, "Recall": 0.5990, "F1-score": 0.6297}
CONFUSION = [[10274, 2797], [3695, 5519]]
N_TOTAL, N_TRAIN, N_TEST = 74283, 51998, 22285
CLASS_COUNTS = {0: 43570, 1: 30713}


# ===========================================================================
# 2. LOGIC (no Streamlit calls, so it can be tested on its own)
# ===========================================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_row(vals: dict, feature_order) -> pd.DataFrame:
    """Turn the 23 form answers into the 29 model inputs, in training order."""
    row = dict.fromkeys(feature_order, 0)
    for key, _label, feature, spec, _help in FIELDS:
        v = vals[key]
        if feature is None:
            row[spec[v]] = 1          # one-hot group
        elif isinstance(spec, dict):
            row[feature] = spec[v]    # dropdown label -> encoded value
        else:
            row[feature] = v          # number box
    return pd.DataFrame([row])[list(feature_order)]


def risk_category(p: float):
    if p < 0.33:
        return "Lower", "success"
    if p < 0.66:
        return "Moderate", "warning"
    return "Higher", "error"


def influence(model, X: pd.DataFrame) -> pd.Series:
    """Each feature's push on the log-odds: standardised value x coefficient."""
    z = model.named_steps["standardscaler"].transform(X)[0]
    coef = model.named_steps["logisticregression"].coef_[0]
    return pd.Series(z * coef, index=X.columns).sort_values(key=abs, ascending=False)


def coefficient_table(model) -> pd.DataFrame:
    coef = model.named_steps["logisticregression"].coef_[0]
    df = pd.DataFrame({"Feature": model.feature_names_in_, "Coefficient": coef})
    df["Direction"] = df["Coefficient"].apply(
        lambda c: "Near zero" if abs(c) < 0.01 else ("Higher → more class 1" if c > 0 else "Higher → more class 0"))
    return df.reindex(df["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ===========================================================================
# 3. SHARED PIECES
# ===========================================================================
NOTICE = ("**Experimental machine-learning project. Not a diagnostic tool.** Its output must not "
          "replace professional medical assessment. If you are worried about memory or brain "
          "health, please speak with a doctor.")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;700&display=swap');
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; }
.hero { border-left: 4px solid #2F6F8F; padding: 0.3rem 0 0.3rem 1.2rem; margin: 0.5rem 0 1rem; }
.hero h1 { font-size: 2.2rem; line-height: 1.2; margin-bottom: 0.6rem; }
.hero p { max-width: 62ch; font-size: 1.08rem; }
</style>
"""


def load_or_stop():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH.name}. Put it in the same folder as alzheimers.py.")
        st.stop()
    return load_model()


# ===========================================================================
# 4. PAGES
# ===========================================================================
def home_page():
    st.markdown(
        '<div class="hero"><h1>Understanding Alzheimer’s Disease Through Data and Machine Learning</h1>'
        "<p>An educational project that shows how a machine-learning model uses everyday health, "
        "lifestyle and background information to produce an experimental estimate. "
        "Explore how it works, what it can do, and where it falls short.</p></div>",
        unsafe_allow_html=True,
    )
    b1, b2, _ = st.columns([1, 1.2, 2])
    if b1.button("Start Assessment", type="primary", use_container_width=True):
        st.switch_page(PREDICT)
    if b2.button("Learn About the Model", use_container_width=True):
        st.switch_page(INSIGHTS)

    st.subheader("How it works")
    steps = [("1. Enter information", "Answer 23 short questions about background, health history and lifestyle."),
             ("2. The model processes it", "Your answers are converted into 29 numeric inputs and passed to a trained logistic regression model."),
             ("3. View the experimental result", "See the model's output with a plain-language explanation of what it does and does not mean.")]
    for col, (title, text) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)

    st.subheader("Important notice")
    st.warning(NOTICE)

    st.subheader("Project overview")
    st.write(
        f"The model was trained on a tabular dataset of {N_TOTAL:,} records with 29 input features "
        "(after converting categories to numbers) and a yes/no Alzheimer's diagnosis column as the "
        "target. A standard scaler followed by logistic regression was fitted on 70% of the records "
        "and evaluated on the remaining 30%. The Model Insights and About pages give the details."
    )


def predict_page():
    model = load_or_stop()
    if "result" in st.session_state:
        return show_result(model)

    st.title("Alzheimer’s Prediction")
    st.warning(NOTICE)
    st.write("Answer every question, then submit. All fields are required.")
    with st.expander("What do some of these terms mean?"):
        st.markdown(
            "- **APOE-ε4:** a gene variant that is more common in people with Alzheimer's. "
            "Only answer Yes if you have had a genetic test showing it.\n"
            "- **Cognitive test score:** a numeric result from a memory and thinking test, on the dataset's 30 to 99 scale.\n"
            "- **BMI:** weight (kg) divided by height (m) squared.\n"
            "- Hover over the **?** next to a question for more detail."
        )

    # Restore earlier answers (Streamlit forgets widget values on pages you leave)
    saved = st.session_state.get("saved", {})
    for key, *_ in FIELDS:
        st.session_state.setdefault(f"f_{key}", saved.get(key))

    def reset_form():
        for key, *_ in FIELDS:
            st.session_state[f"f_{key}"] = None
        st.session_state["saved"] = {}

    with st.form("assessment"):
        for title, fields in SECTIONS:
            st.subheader(title)
            cols = st.columns(2)
            for i, (key, label, _feat, spec, help_) in enumerate(fields):
                with cols[i % 2]:
                    k = f"f_{key}"
                    if isinstance(spec, dict):
                        st.selectbox(label, list(spec), index=None, placeholder="Select an option",
                                     key=k, help=help_ or None)
                    else:
                        lo, hi, step = spec
                        st.number_input(label, min_value=lo, max_value=hi, value=None, step=step,
                                        format="%d" if isinstance(step, int) else "%.1f",
                                        placeholder=f"{lo} to {hi}", key=k, help=help_ or None)
        c1, c2, _ = st.columns([1.3, 1, 2])
        submitted = c1.form_submit_button("Submit for prediction", type="primary", use_container_width=True)
        c2.form_submit_button("Reset form", on_click=reset_form, use_container_width=True)

    if submitted:
        vals = {key: st.session_state[f"f_{key}"] for key, *_ in FIELDS}
        missing = [label for key, label, *_ in FIELDS if vals[key] is None]
        if missing:
            st.error("Please complete these fields before submitting: " + ", ".join(missing) +
                     ". Your other answers have been kept. Numbers must stay within the range shown in each box.")
        else:
            with st.spinner("Running the model…"):
                X = build_row(vals, model.feature_names_in_)
                prob = float(model.predict_proba(X)[0, 1])
            st.session_state["saved"] = vals
            st.session_state["result"] = {"X": X, "prob": prob}
            st.rerun()


def show_result(model):
    res = st.session_state["result"]
    X, p1 = res["X"], res["prob"]
    p0 = 1 - p1
    predicted = int(p1 >= 0.5)
    label, level = risk_category(p1)

    st.title("Prediction result")

    st.subheader("A. Prediction summary")
    getattr(st, level)(f"**{label} estimated risk** according to this demo model "
                       f"(model output: about {p1:.0%}).")
    st.progress(p1)
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted class", f"Class {predicted}")
    m2.metric("Probability of class 0", f"{p0:.1%}")
    m3.metric("Probability of class 1", f"{p1:.1%}")
    st.markdown(
        "- **Class 0:** the dataset records no Alzheimer's diagnosis for this kind of profile.\n"
        "- **Class 1:** the dataset records an Alzheimer's diagnosis for this kind of profile."
    )
    st.caption("Illustrative values only. Class 1 is not a confirmed diagnosis or an individual's clinical risk. "
               "The class meanings come from the dataset's “Alzheimer’s Diagnosis” column (0 and 1).")

    st.subheader("B. Interpretation")
    st.write("This output was produced by a logistic regression model that compares the submitted answers "
             "with patterns in a public dataset. The predicted class is class 1 when the model's probability "
             "for class 1 is 50% or higher. It describes how similar your answers are to records labelled "
             "class 1 in that dataset. It says nothing about your own health.")
    with st.expander("Which answers moved the result most? (for learning)"):
        st.write("Positive values pushed the output up, negative values pushed it down. "
                 "These are statistical contributions, not medical causes.")
        st.bar_chart(influence(model, X).head(6).rename("influence"))
    with st.expander("Show the exact 29 values sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))

    st.subheader("C. Important limitations")
    st.markdown(
        "- This is an experimental project and has not been clinically validated.\n"
        "- The model is right about 71% of the time on held-out data from the same dataset, and it misses "
        "roughly 4 in 10 class 1 cases. It will make mistakes in both directions.\n"
        "- The dataset may not represent you, and real-world risk depends on many factors the model cannot see.\n"
        "- Do not use this result to make health decisions."
    )

    st.subheader("D. What next?")
    a1, a2, a3 = st.columns(3)
    if a1.button("Make another prediction", type="primary", use_container_width=True):
        del st.session_state["result"]
        st.rerun()
    if a2.button("Return home", use_container_width=True):
        st.switch_page(HOME)
    if a3.button("About the model", use_container_width=True):
        st.switch_page(INSIGHTS)


def insights_page():
    model = load_or_stop()
    st.title("Model Insights")

    st.subheader("1. Model information")
    st.write("The model is a scikit-learn pipeline with two steps: a **StandardScaler**, which puts every "
             "input on a comparable scale, followed by a **Logistic Regression** classifier. "
             "It takes **29 input features** and outputs a probability for each of two classes.")

    st.subheader("2. Evaluation metrics")
    cols = st.columns(4)
    for col, (name, val) in zip(cols, METRICS.items()):
        col.metric(name, f"{val:.1%}" if name == "Accuracy" else f"{val:.2f}")
    st.caption(f"Measured on the {N_TEST:,} held-out test records, with class 1 as the positive class. "
               "Reproduced with the saved model on a stratified 70/30 split (random state 42), matching "
               "the reported ≈71% accuracy and 0.63 F1-score. This assumes the model was trained on the 70% partition.")
    cm = pd.DataFrame(CONFUSION, index=["Actual class 0", "Actual class 1"],
                      columns=["Predicted class 0", "Predicted class 1"])
    st.write("**Confusion matrix** (number of test records)")
    st.dataframe(cm)
    st.write("Precision is how often a class 1 prediction is correct. Recall is the share of real class 1 "
             "records the model finds. Recall below 60% means many class 1 records are missed.")

    st.subheader("3. Feature coefficients")
    ct = coefficient_table(model)
    st.bar_chart(ct.set_index("Feature")["Coefficient"], horizontal=True, height=620)
    st.dataframe(ct, hide_index=True, use_container_width=True)
    top = ", ".join(ct["Feature"].head(3))
    st.info(
        f"**How to read this.** Inputs are standardised, so each coefficient is the change in the model's "
        f"log-odds of class 1 for a one-standard-deviation increase in that input. The largest are {top}; "
        "most others are close to zero, so the model barely uses them. Coefficients show association within this "
        "dataset, **not causation**, and their size is **not a measure of clinical importance**. "
        "Encoded categories (like Low=0, High=2) and the one-hot groups also make single coefficients hard to read on their own."
    )

    st.subheader("4. Dataset information")
    info = pd.DataFrame({
        "Item": ["Records", "Input features", "Target classes", "Class 0 records", "Class 1 records",
                 "Training / test split", "Stratified", "Random state", "Training / test records"],
        "Value": [f"{N_TOTAL:,}", "29", "0 and 1", f"{CLASS_COUNTS[0]:,}", f"{CLASS_COUNTS[1]:,}",
                  "70% / 30%", "Yes", "42", f"{N_TRAIN:,} / {N_TEST:,}"],
    })
    st.dataframe(info, hide_index=True, use_container_width=True)


def about_page():
    st.title("About this project")
    sections = {
        "Background of Alzheimer's disease":
            "Alzheimer's disease is the most common cause of dementia, a gradual loss of memory, thinking and "
            "everyday functioning. Age is the strongest known risk factor, and genetics, cardiovascular health, "
            "education and lifestyle are also studied. There is currently no cure, which is why research into "
            "early signals and risk factors matters.",
        "Motivation":
            "The project shows how a machine-learning workflow, from data to a working web page, can be built and "
            "explained openly. It is meant for learning about modelling and about the care needed when "
            "health-related predictions are shown to people.",
        "Project objectives":
            "To train a transparent baseline classifier; to present it through an accessible interface; to report "
            "honestly how well it performs; and to make its limits clear to non-specialists.",
        "Dataset description and source":
            f"A tabular dataset of {N_TOTAL:,} records, each with 29 numeric inputs after encoding and a binary "
            "Alzheimer's diagnosis label. The inputs cover demographics, medical history, lifestyle, environment "
            "and a cognitive test score. It was supplied with the project and appears to be synthetic or "
            "simulated, which would explain why most inputs have almost no effect on the model.",
        "Data preprocessing":
            "The supplied file was already processed: categories were converted to numbers (ordered levels such "
            "as Low, Medium and High became 0, 1 and 2) and three groups (marital, smoking and employment status) "
            "were one-hot encoded into separate 0/1 columns. Inside the model, a standard scaler then "
            "rescales every input.",
        "Model selection":
            "Logistic regression was chosen as an interpretable baseline: each input has one coefficient that can "
            "be inspected, and the output is a probability.",
        "Model training":
            "The scaler and classifier were fitted together as one pipeline on a stratified 70% training "
            "partition (random state 42), so both classes keep their share in each partition.",
        "Evaluation methodology":
            "The remaining 30% was held back and used only for testing. Accuracy, precision, recall, F1-score "
            "and a confusion matrix are reported for class 1. Evaluation used the same dataset, so it says "
            "little about performance on new populations.",
        "Ethical considerations":
            "Predictions about serious illness can cause distress and false reassurance. This site therefore "
            "presents itself as an experiment, avoids diagnostic language, and does not save the answers you "
            "enter. Please do not enter information you are not comfortable sharing.",
        "Limitations and future improvements":
            "The model is simple, the data may not reflect real patients, and recall is modest. Future work "
            "could include external validation on real cohorts, probability calibration, comparison with other "
            "algorithms, fairness checks across groups, and uncertainty ranges on predictions.",
    }
    for heading, text in sections.items():
        st.subheader(heading)
        st.write(text)


# ===========================================================================
# 5. APP ENTRY POINT: shared navigation + page registry
# ===========================================================================
st.set_page_config(page_title="Alzheimer’s ML Project", page_icon="🧠", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

HOME = st.Page(home_page, title="Home", icon="🏠", url_path="home", default=True)
PREDICT = st.Page(predict_page, title="Alzheimer's Prediction", icon="🧪", url_path="prediction")
INSIGHTS = st.Page(insights_page, title="Model Insights", icon="📊", url_path="insights")
ABOUT = st.Page(about_page, title="About", icon="ℹ️", url_path="about")

try:
    nav = st.navigation([HOME, PREDICT, INSIGHTS, ABOUT], position="top")
except TypeError:  # older Streamlit: fall back to the sidebar menu
    nav = st.navigation([HOME, PREDICT, INSIGHTS, ABOUT])

st.markdown("**🧠 Alzheimer’s Disease Prediction Using Machine Learning**")
nav.run()
