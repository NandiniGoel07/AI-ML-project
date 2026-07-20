import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Intelligent NIDS with XAI", layout="wide")

st.title("🛡️ Intelligent Network Intrusion Detection System")
st.markdown("Predicts network traffic as **Normal / DoS / Probe / R2L / U2R**, with SHAP-based explanations.")
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, scaler, label_encoder, feature_names, explainer

model, scaler, label_encoder, feature_names, explainer = load_artifacts()
class_names = label_encoder.classes_
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "How do you want to provide traffic data?",
    ["Upload CSV", "Try a sample row"]
)

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Random Forest", "Decision Tree", "XGBoost"]
)

model_paths = {
    "Random Forest": "models/random_forest_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

if model_choice != "Random Forest":
    model = joblib.load(model_paths[model_choice])

input_df = None
already_scaled = False

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with network traffic features", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df.columns = input_df.columns.str.strip()
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())

already_scaled = False

if input_method == "Try a sample row":
    sample_data = pd.read_csv("app/sample_traffic.csv")
    row_num = st.selectbox("Pick a sample traffic row to test:", sample_data.index)
    input_df = sample_data.loc[[row_num]]
    st.write("Selected row:")
    st.dataframe(input_df)
    already_scaled = True

if input_df is not None:
    if st.button("🔍 Predict"):
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            st.error(f"Uploaded data is missing columns: {missing_cols}")
        else:
           X_input = input_df[feature_names]
           if already_scaled:
            X_input_scaled = X_input.values
           else:
            X_input_scaled = scaler.transform(X_input)

           predictions = model.predict(X_input_scaled)
           predicted_labels = label_encoder.inverse_transform(predictions)

           st.subheader("Prediction Results")
           result_df = input_df.copy()
           result_df['Predicted_Category'] = predicted_labels
           st.dataframe(result_df[['Predicted_Category']].join(result_df.drop(columns=['Predicted_Category'])))   
           st.subheader("🔎 Why did the model predict this? (SHAP Explanation)")   
           explain_idx = 0
           single_row_scaled = X_input_scaled[explain_idx:explain_idx+1]
           pred_class_idx = predictions[explain_idx]   
           shap_values_single = explainer.shap_values(single_row_scaled)   
           if isinstance(shap_values_single, list):
               row_shap = shap_values_single[pred_class_idx][0, :]
           else:
               row_shap = shap_values_single[0, :, pred_class_idx]   
           shap_df = pd.DataFrame({
               'Feature': feature_names,
               'SHAP Value': row_shap
           })
           shap_df['Abs Value'] = shap_df['SHAP Value'].abs()
           top_features = shap_df.sort_values('Abs Value', ascending=False).head(10)   
           fig, ax = plt.subplots(figsize=(8, 5))
           colors = ['crimson' if v > 0 else 'steelblue' for v in top_features['SHAP Value']]
           ax.barh(top_features['Feature'], top_features['SHAP Value'], color=colors)
           ax.set_xlabel('SHAP value (impact on prediction)')
           ax.set_title(f'Top features influencing prediction: {predicted_labels[explain_idx]}')
           ax.invert_yaxis()
           st.pyplot(fig)   
           st.markdown("""
           **How to read this:** Red bars pushed the prediction *toward* the predicted class.
           Blue bars pushed *away* from it. Longer bars = stronger influence.
           """)   
           st.sidebar.markdown("---")

if st.sidebar.checkbox("Show model comparison metrics"):
    comparison_df = pd.read_csv('reports/model_comparison.csv', index_col=0)
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(comparison_df)
    st.bar_chart(comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']])