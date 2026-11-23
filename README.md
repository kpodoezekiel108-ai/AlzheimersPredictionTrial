# Alzheimer's Risk Estimator (Educational Demo)

A Streamlit app that runs a trained scikit-learn model (StandardScaler + logistic regression)
on user-supplied characteristics. **Educational use only. Not a medical tool or diagnosis.**

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – the web app (label-to-code mappings are at the top)
- `AlzheimersPredictionModel.joblib` – trained model (scikit-learn 1.6.1)
- `requirements.txt` – dependencies
