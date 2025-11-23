import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
    }
    .safe-alert {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #44ff44;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# Header
st.markdown('<p class="main-header">🔒 Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-powered transaction analysis with explainable AI</p>', unsafe_allow_html=True)

st.divider()

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Transaction Details")
    
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
        help="Select the type of transaction"
    )
    
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Enter the transaction amount"
    )
    
    st.markdown("#### Sender Account")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        oldbalanceOrg = st.number_input(
            "Old Balance",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key="old_sender"
        )
    with col_s2:
        newbalanceOrig = st.number_input(
            "New Balance",
            min_value=0.0,
            value=9000.0,
            step=100.0,
            key="new_sender"
        )
    
    st.markdown("#### Receiver Account")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        oldbalanceDist = st.number_input(
            "Old Balance",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key="old_receiver"
        )
    with col_r2:
        newbalanceDist = st.number_input(
            "New Balance",
            min_value=0.0,
            value=11000.0,
            step=100.0,
            key="new_receiver"
        )

with col2:
    st.subheader("📊 Analysis Results")
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDist,
            "newbalanceDest": newbalanceDist
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability if available
        try:
            prediction_proba = model.predict_proba(input_data)[0]
            fraud_probability = prediction_proba[1] * 100
            safe_probability = prediction_proba[0] * 100
        except AttributeError:
            fraud_probability = 100 if prediction == 1 else 0
            safe_probability = 0 if prediction == 1 else 100
        
        # Display prediction
        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown(f"""
                <div class="fraud-alert">
                    <h2 style="color: #cc0000; margin: 0;">⚠️ FRAUD DETECTED</h2>
                    <p style="margin: 0.5rem 0 0 0;">This transaction shows suspicious patterns</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="safe-alert">
                    <h2 style="color: #00cc00; margin: 0;">✅ LEGITIMATE</h2>
                    <p style="margin: 0.5rem 0 0 0;">This transaction appears to be safe</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Fraud Score")
        
        # Display probabilities with metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Fraud Probability", f"{fraud_probability:.2f}%", 
                     delta=None, delta_color="inverse")
        with metric_col2:
            st.metric("Safe Probability", f"{safe_probability:.2f}%",
                     delta=None, delta_color="normal")
        
        # Progress bar for fraud score
        st.progress(fraud_probability / 100)
        
        # Risk level
        if fraud_probability >= 75:
            risk_level = "🔴 Critical Risk"
            risk_color = "#cc0000"
        elif fraud_probability >= 50:
            risk_level = "🟠 High Risk"
            risk_color = "#ff8800"
        elif fraud_probability >= 25:
            risk_level = "🟡 Medium Risk"
            risk_color = "#ffbb00"
        else:
            risk_level = "🟢 Low Risk"
            risk_color = "#00cc00"
        
        st.markdown(f"**Risk Level:** <span style='color: {risk_color}; font-weight: bold;'>{risk_level}</span>", 
                   unsafe_allow_html=True)
        
        # SHAP Explanation
        st.divider()
        st.subheader("🧠 Model Explanation (SHAP)")
        
        with st.spinner("Generating explanation..."):
            try:
                # Extract the actual classifier from the pipeline
                if hasattr(model, 'named_steps'):
                    # Get all steps except the last one (classifier)
                    steps = list(model.named_steps.keys())
                    
                    # Find the classifier step (usually the last one)
                    classifier_name = steps[-1]
                    clf = model.named_steps[classifier_name]
                    
                    # Transform the data through all preprocessing steps
                    X_transformed = input_data.copy()
                    for step_name in steps[:-1]:
                        X_transformed = model.named_steps[step_name].transform(X_transformed)
                    
                    # Get feature names after transformation
                    if hasattr(model.named_steps[steps[-2]], 'get_feature_names_out'):
                        feature_names = list(model.named_steps[steps[-2]].get_feature_names_out())
                    else:
                        feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]
                else:
                    clf = model
                    X_transformed = input_data.values
                    feature_names = input_data.columns.tolist()
                
                # Convert to numpy array if needed
                if hasattr(X_transformed, 'toarray'):
                    X_transformed = X_transformed.toarray()
                elif isinstance(X_transformed, pd.DataFrame):
                    X_transformed = X_transformed.values
                
                # Determine the type of explainer to use
                model_type = type(clf).__name__
                
                if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type or 'LightGBM' in model_type or 'CatBoost' in model_type:
                    # Use TreeExplainer for tree-based models
                    explainer = shap.TreeExplainer(clf)
                else:
                    # Use LinearExplainer for linear models (Logistic Regression, etc.)
                    explainer = shap.LinearExplainer(clf, X_transformed, feature_dependence="independent")
                
                shap_values = explainer.shap_values(X_transformed)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_fraud = shap_values[1]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_values_fraud = shap_values
                    base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
                
                # Create waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values_fraud[0] if len(shap_values_fraud.shape) > 1 else shap_values_fraud,
                        base_values=float(base_value),
                        data=X_transformed[0],
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                **How to read this chart:**
                - 🔴 Red bars push the prediction toward FRAUD
                - 🔵 Blue bars push the prediction toward LEGITIMATE
                - Larger bars have more influence on the prediction
                - E[f(x)] is the baseline prediction (average)
                - f(x) is the final prediction for this transaction
                """)
                
            except Exception as e:
                st.warning(f"⚠️ SHAP explanation not available for this model type")
                
                # Provide alternative feature importance
                st.markdown("### 📊 Feature Coefficients (Alternative Explanation)")
                try:
                    if hasattr(clf, 'coef_'):
                        coef_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_
                        })
                        coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
                        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['red' if x > 0 else 'blue' for x in coef_df['Coefficient']]
                        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
                        ax.set_xlabel('Coefficient Value')
                        ax.set_title('Top 10 Most Important Features')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.info("""
                        **Feature Coefficients:**
                        - 🔴 Positive values increase fraud probability
                        - 🔵 Negative values decrease fraud probability
                        - Larger absolute values = stronger influence
                        """)
                except:
                    st.info("Feature importance visualization not available for this model configuration.")

# Footer with information
st.divider()
with st.expander("ℹ️ About this System"):
    st.markdown("""
    ### Fraud Detection System
    
    This system uses machine learning to identify potentially fraudulent transactions based on:
    - **Transaction type**: Different types have different risk profiles
    - **Amount**: Unusual amounts may indicate fraud
    - **Balance changes**: Inconsistent balance updates are suspicious
    - **Account behavior**: Patterns in sender and receiver accounts
    
    ### Features:
    - ✅ Real-time fraud prediction
    - ✅ Probability scores for risk assessment
    - ✅ SHAP explanations for model transparency
    - ✅ Interactive and user-friendly interface
    
    **Note:** This is a predictive tool. Always verify suspicious transactions through additional means.
    """)