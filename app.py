"""
Obesity Level Classification - Streamlit Application
=====================================================
BITS WILP M.Tech AIML - Machine Learning Assignment 2

This application demonstrates multiple classification models for predicting
obesity levels based on eating habits and physical condition.

Features:
- Dataset upload (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization
- Classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}
    st.session_state.scaler = None
    st.session_state.label_encoders = None
    st.session_state.feature_names = None
    st.session_state.class_labels = None


@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'KNN': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    base_path = 'model/trained_models/'
    
    for name, filename in model_files.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
    
    # Load preprocessing objects
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl')) if os.path.exists(os.path.join(base_path, 'scaler.pkl')) else None
    label_encoders = joblib.load(os.path.join(base_path, 'label_encoders.pkl')) if os.path.exists(os.path.join(base_path, 'label_encoders.pkl')) else None
    feature_names = joblib.load(os.path.join(base_path, 'feature_names.pkl')) if os.path.exists(os.path.join(base_path, 'feature_names.pkl')) else None
    class_labels = joblib.load(os.path.join(base_path, 'class_labels.pkl')) if os.path.exists(os.path.join(base_path, 'class_labels.pkl')) else None
    
    return models, scaler, label_encoders, feature_names, class_labels


@st.cache_data
def load_results():
    """Load pre-computed model results"""
    results_path = 'model/trained_models/model_results.csv'
    if os.path.exists(results_path):
        return pd.read_csv(results_path, index_col=0)
    return None


def preprocess_data(df, label_encoders, scaler, feature_names):
    """Preprocess uploaded data for prediction"""
    df_processed = df.copy()
    
    # Identify target column if present
    target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else None
    y = None
    
    if target_col:
        y = df_processed[target_col].copy()
        df_processed = df_processed.drop(target_col, axis=1)
    
    # Label encode categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in label_encoders:
            try:
                df_processed[col] = label_encoders[col].transform(df_processed[col])
            except ValueError:
                # Handle unseen labels
                le = label_encoders[col]
                df_processed[col] = df_processed[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # Ensure columns are in correct order
    if feature_names:
        df_processed = df_processed[feature_names]
    
    # Scale features
    if scaler:
        X_scaled = scaler.transform(df_processed)
    else:
        X_scaled = df_processed.values
    
    # Encode target if present
    y_encoded = None
    if y is not None and 'NObeyesdad' in label_encoders:
        try:
            y_encoded = label_encoders['NObeyesdad'].transform(y)
        except:
            y_encoded = None
    
    return X_scaled, y_encoded, y


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC Score
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_labels):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🏋️ Obesity Level Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-class Classification using Machine Learning Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title(" Navigation")
    
    # Try to load models
    try:
        models, scaler, label_encoders, feature_names, class_labels = load_models()
        models_available = len(models) > 0
    except:
        models_available = False
        models, scaler, label_encoders, feature_names, class_labels = {}, None, None, None, None
    
    # Load pre-computed results
    results_df = load_results()
    
    # Sidebar - Model Selection
    st.sidebar.header("🤖 Model Selection")
    
    if models_available:
        selected_model = st.sidebar.selectbox(
            "Choose a Classification Model:",
            list(models.keys()),
            help="Select a model to use for predictions"
        )
    else:
        selected_model = st.sidebar.selectbox(
            "Choose a Classification Model:",
            ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
            help="Models will be used once data is uploaded"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload & Prediction", "📈 Model Metrics", "🔍 Model Comparison", "ℹ️ About"])
    
    # Tab 1: Data Upload & Prediction
    with tab1:
        st.header("Upload Test Data")
        
        st.info("""
        **Instructions:**
        1. Upload a CSV file containing test data
        2. The file should have the same features as the training data
        3. Optionally include 'NObeyesdad' column for evaluation
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your test dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f" File uploaded successfully! Shape: {df.shape}")
                
                # Display data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check if models are available
                if not models_available:
                    st.warning(" Models not loaded. Please ensure model files are in the correct directory.")
                else:
                    # Preprocess data
                    X_scaled, y_encoded, y_original = preprocess_data(
                        df, label_encoders, scaler, feature_names
                    )
                    
                    # Get selected model
                    model = models[selected_model]
                    
                    # Make predictions
                    st.subheader(f"🎯 Predictions using {selected_model}")
                    
                    y_pred = model.predict(X_scaled)
                    y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
                    
                    # Decode predictions
                    if class_labels and label_encoders:
                        y_pred_labels = label_encoders['NObeyesdad'].inverse_transform(y_pred)
                    else:
                        y_pred_labels = y_pred
                    
                    # Display predictions
                    predictions_df = df.copy()
                    predictions_df['Predicted_Obesity_Level'] = y_pred_labels
                    
                    if y_pred_proba is not None:
                        max_proba = np.max(y_pred_proba, axis=1)
                        predictions_df['Confidence'] = [f"{p:.2%}" for p in max_proba]
                    
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Download predictions
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # If true labels available, show evaluation
                    if y_encoded is not None:
                        st.subheader(" Evaluation Metrics")
                        
                        metrics = calculate_metrics(y_encoded, y_pred, y_pred_proba)
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        with col2:
                            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                        with col3:
                            st.metric("Precision", f"{metrics['Precision']:.4f}")
                        with col4:
                            st.metric("Recall", f"{metrics['Recall']:.4f}")
                        with col5:
                            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        with col6:
                            st.metric("MCC", f"{metrics['MCC']:.4f}")
                        
                        # Confusion Matrix
                        st.subheader("📉 Confusion Matrix")
                        fig = plot_confusion_matrix(y_encoded, y_pred, class_labels)
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.subheader(" Classification Report")
                        report = classification_report(y_encoded, y_pred, 
                                                       target_names=class_labels,
                                                       output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Tab 2: Model Metrics
    with tab2:
        st.header("📈 Individual Model Metrics")
        
        if results_df is not None:
            st.subheader(f"Metrics for: {selected_model}")
            
            if selected_model in results_df.index:
                model_metrics = results_df.loc[selected_model]
                
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
                with col2:
                    st.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
                with col3:
                    st.metric("Precision", f"{model_metrics['Precision']:.4f}")
                with col4:
                    st.metric("Recall", f"{model_metrics['Recall']:.4f}")
                with col5:
                    st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
                with col6:
                    st.metric("MCC", f"{model_metrics['MCC']:.4f}")
                
                # Radar chart for model metrics
                st.subheader("📊 Metrics Visualization")
                
                metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
                values = [model_metrics[m] for m in metrics_names]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
                bars = ax.bar(metrics_names, values, color=colors, edgecolor='black')
                
                ax.set_ylim(0, 1.1)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title(f'{selected_model} - Performance Metrics', fontsize=14, fontweight='bold')
                
                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Model results not available. Please run the training notebook first.")
    
    # Tab 3: Model Comparison
    with tab3:
        st.header("🔍 Model Comparison")
        
        if results_df is not None:
            # Display comparison table
            st.subheader(" Comparison Table")
            st.dataframe(
                results_df.style.highlight_max(axis=0, color='lightgreen')
                          .format("{:.4f}"),
                use_container_width=True
            )
            
            # Comparison charts
            st.subheader("📈 Visual Comparison")
            
            metric_to_compare = st.selectbox(
                "Select metric to compare:",
                ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            models_list = results_df.index.tolist()
            values = results_df[metric_to_compare].values
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models_list)))
            bars = ax.barh(models_list, values, color=colors, edgecolor='black')
            
            ax.set_xlabel(metric_to_compare, fontsize=12)
            ax.set_title(f'{metric_to_compare} Comparison Across Models', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model summary
            st.subheader(" Best Model Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc = results_df['Accuracy'].idxmax()
                st.info(f"**Best Accuracy**\n\n{best_acc}\n\n{results_df.loc[best_acc, 'Accuracy']:.4f}")
            
            with col2:
                best_f1 = results_df['F1'].idxmax()
                st.info(f"**Best F1 Score**\n\n{best_f1}\n\n{results_df.loc[best_f1, 'F1']:.4f}")
            
            with col3:
                best_auc = results_df['AUC'].idxmax()
                st.info(f"**Best AUC**\n\n{best_auc}\n\n{results_df.loc[best_auc, 'AUC']:.4f}")
        else:
            st.warning("Model results not available. Please run the training notebook first.")
    
    # Tab 4: About
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ###  Dataset Information
        
        **Dataset Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition
        
        **Source:** UCI Machine Learning Repository
        
        **Description:** This dataset contains data for estimating obesity levels in individuals from 
        Mexico, Peru, and Colombia based on their eating habits and physical condition.
        
        - **Features:** 16 input features
        - **Instances:** 2,111 records
        - **Target Classes:** 7 obesity levels
          - Insufficient Weight
          - Normal Weight
          - Overweight Level I
          - Overweight Level II
          - Obesity Type I
          - Obesity Type II
          - Obesity Type III
        
        ### 🤖 Models Implemented
        
        | Model | Type | Description |
        |-------|------|-------------|
        | Logistic Regression | Linear | Baseline linear classifier |
        | Decision Tree | Tree-based | Non-linear classifier with interpretable rules |
        | KNN | Instance-based | Distance-based classifier |
        | Naive Bayes | Probabilistic | Gaussian probability-based classifier |
        | Random Forest | Ensemble | Bagging ensemble of decision trees |
        | XGBoost | Ensemble | Gradient boosting ensemble |
        
        ### 📊 Evaluation Metrics
        
        - **Accuracy:** Overall correctness of predictions
        - **AUC (ROC):** Area under the ROC curve (multi-class weighted)
        - **Precision:** Positive predictive value
        - **Recall:** Sensitivity / True positive rate
        - **F1 Score:** Harmonic mean of precision and recall
        - **MCC:** Matthews Correlation Coefficient
        
        """)


if __name__ == "__main__":
    main()
