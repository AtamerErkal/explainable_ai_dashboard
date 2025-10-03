import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            f1_score, recall_score, precision_score)
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Explainable AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Headers - inherit theme color */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: inherit !important;
    }
    
    /* Paragraphs - inherit theme color */
    .main p {
        color: inherit !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Gradient boxes - ALWAYS white text */
    .description-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .description-box,
    .description-box *,
    .description-box h3,
    .description-box p,
    .description-box strong,
    .description-box b {
        color: white !important;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box,
    .metric-box *,
    .metric-box h2,
    .metric-box h3 {
        color: white !important;
    }
    
    .data-summary-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .data-summary-box,
    .data-summary-box *,
    .data-summary-box h2,
    .data-summary-box h3 {
        color: white !important;
    }
    
    /* Light boxes - ALWAYS dark text */
    .info-box {
        background: rgba(240, 242, 246, 0.95);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box,
    .info-box *,
    .info-box b,
    .info-box strong {
        color: #1a1a1a !important;
    }
    
    .explanation-box {
        background: rgba(232, 244, 248, 0.95);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .explanation-box,
    .explanation-box *,
    .explanation-box b,
    .explanation-box strong {
        color: #1a1a1a !important;
    }
    
    /* Welcome boxes - dark text */
    div[style*='background: #f0f2f6'] h2,
    div[style*='background: #f0f2f6'] h3,
    div[style*='background: #f0f2f6'] p {
        color: #1a1a1a !important;
    }
    
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Alert messages - dark text */
    .stAlert,
    .stAlert *,
    .stAlert b,
    .stAlert strong {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">🔍 Explainable AI Dashboard</p>', unsafe_allow_html=True)

# Description
st.markdown("""
<div class="description-box">
    <h3>🎯 What This Dashboard Does</h3>
    <p style='font-size: 1.1rem; margin-top: 1rem;'>
        This interactive dashboard helps you understand <strong>how and why</strong> your machine learning model makes predictions.
        Upload your data, train a model, and explore feature importance, SHAP values, and LIME explanations to gain deep insights 
        into your model's decision-making process. Compare different explainability methods side-by-side to build trust in your AI system.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'X_test_original' not in st.session_state:
    st.session_state.X_test_original = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Select CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df.copy()
        st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Target variable selection
        st.markdown("### 🎯 Target Variable")
        target_col = st.selectbox("Select target column", df.columns.tolist())
        
        # Feature selection
        st.markdown("### 📊 Features")
        available_features = [col for col in df.columns if col != target_col]
        feature_cols = st.multiselect(
            "Select feature columns",
            available_features,
            default=available_features[:min(10, len(available_features))]
        )
        
        if len(feature_cols) > 0:
            st.markdown("---")
            
            # Model selection
            st.markdown("### 🤖 Model Selection")
            model_choice = st.selectbox(
                "Algorithm",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"]
            )
            
            # Hyperparameters
            with st.expander("🔧 Hyperparameters"):
                if model_choice == "Random Forest":
                    n_estimators = st.slider("n_estimators", 10, 200, 100, 10)
                    max_depth = st.slider("max_depth", 3, 20, 10)
                elif model_choice == "Gradient Boosting":
                    n_estimators = st.slider("n_estimators", 10, 200, 100, 10)
                    learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
                else:
                    max_iter = st.slider("max_iter", 100, 1000, 200, 100)
            
            # Train/test split
            test_size = st.slider("Test set ratio (%)", 10, 50, 20, 5) / 100
            random_state = st.number_input("Random state", 0, 100, 42)
            
            st.markdown("---")
            
            # Train button
            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                with st.spinner("🔄 Training model..."):
                    try:
                        # Prepare data
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()
                        
                        # Store original X for later display
                        X_original = X.copy()
                        
                        # Handle missing values
                        X = X.fillna(X.mean(numeric_only=True))
                        for col in X.select_dtypes(include=['object']).columns:
                            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
                        
                        # Encode categorical variables
                        le_dict = {}
                        for col in X.columns:
                            if X[col].dtype == 'object':
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                                le_dict[col] = le
                        
                        # Encode target if categorical
                        target_is_categorical = y.dtype == 'object'
                        if target_is_categorical:
                            le_target = LabelEncoder()
                            y_encoded = le_target.fit_transform(y.astype(str))
                            le_dict['target'] = le_target
                            # Store class names
                            st.session_state.class_names = le_target.classes_.tolist()
                        else:
                            y_encoded = y
                            # For numeric targets, create class names
                            unique_classes = sorted(y.unique())
                            st.session_state.class_names = [f"Class {int(c)}" for c in unique_classes]
                        
                        st.session_state.target_name = target_col
                        
                        # Train/test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=test_size, random_state=random_state
                        )
                        
                        # Also split original data for display
                        X_train_orig, X_test_orig, _, _ = train_test_split(
                            X_original, y_encoded, test_size=test_size, random_state=random_state
                        )
                        
                        # Train model
                        if model_choice == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state,
                                n_jobs=-1
                            )
                        elif model_choice == "Gradient Boosting":
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=random_state
                            )
                        else:
                            model = LogisticRegression(
                                max_iter=max_iter,
                                random_state=random_state
                            )
                        
                        model.fit(X_train, y_train)
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_cols
                        st.session_state.label_encoders = le_dict
                        st.session_state.model_name = model_choice
                        st.session_state.X_test_original = X_test_orig
                        
                        st.success("✅ Model trained successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# Main content area
if st.session_state.model is not None:
    
    # Data Summary Section
    st.markdown("## 📊 Dataset Summary")
    
    if st.session_state.original_df is not None:
        df_summary = st.session_state.original_df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="data-summary-box">
                <h3>📋 Total Rows</h3>
                <h2 style="text-align: center;">{df_summary.shape[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="data-summary-box">
                <h3>📊 Total Columns</h3>
                <h2 style="text-align: center;">{df_summary.shape[1]}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = df_summary.select_dtypes(include=[np.number]).shape[1]
            st.markdown(f"""
            <div class="data-summary-box">
                <h3>🔢 Numeric Features</h3>
                <h2 style="text-align: center;">{numeric_cols}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_cols = df_summary.select_dtypes(include=['object']).shape[1]
            st.markdown(f"""
            <div class="data-summary-box">
                <h3>📝 Categorical Features</h3>
                <h2 style="text-align: center;">{categorical_cols}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed statistics
        with st.expander("📈 Detailed Statistics"):
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("**Numeric Features Summary**")
                st.dataframe(df_summary.describe(), use_container_width=True)
            
            with col_stat2:
                st.markdown("**Missing Values**")
                missing_data = pd.DataFrame({
                    'Column': df_summary.columns,
                    'Missing Count': df_summary.isnull().sum().values,
                    'Missing %': (df_summary.isnull().sum().values / len(df_summary) * 100).round(2)
                })
                missing_data = missing_data[missing_data['Missing Count'] > 0]
                if len(missing_data) > 0:
                    st.dataframe(missing_data, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No missing values found!")
    
    st.markdown("---")
    
    # Model Performance Section
    st.markdown("## 📈 Model Performance Metrics")
    
    # Calculate predictions
    y_pred = st.session_state.model.predict(st.session_state.X_test)
    accuracy = accuracy_score(st.session_state.y_test, y_pred)
    
    # Calculate metrics
    precision = precision_score(st.session_state.y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(st.session_state.y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>🎯 Accuracy</h3>
            <h2 style="text-align: center;">{accuracy:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>🎪 Precision</h3>
            <h2 style="text-align: center;">{precision:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>🔄 Recall</h3>
            <h2 style="text-align: center;">{recall:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>⚖️ F1 Score</h3>
            <h2 style="text-align: center;">{f1:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <h3>📊 Selected Features</h3>
            <h2 style="text-align: center;">{len(st.session_state.feature_names)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Per-Class Metrics for Multi-class
    num_classes = len(st.session_state.class_names) if st.session_state.class_names is not None else 2
    
    if num_classes > 2:
        with st.expander("📊 Per-Class Performance Metrics", expanded=False):
            st.markdown("### Detailed Metrics by Class")
            
            # Get classification report as dict
            report = classification_report(st.session_state.y_test, y_pred, 
                                          target_names=st.session_state.class_names,
                                          output_dict=True, zero_division=0)
            
            # Create DataFrame for per-class metrics
            metrics_data = []
            for class_name in st.session_state.class_names:
                if class_name in report:
                    metrics_data.append({
                        'Class': class_name,
                        'Precision': f"{report[class_name]['precision']:.2%}",
                        'Recall': f"{report[class_name]['recall']:.2%}",
                        'F1-Score': f"{report[class_name]['f1-score']:.2%}",
                        'Support': int(report[class_name]['support'])
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display as styled dataframe
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Visualize per-class metrics
            col_vis1, col_vis2 = st.columns(2)
            
            with col_vis1:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(st.session_state.class_names))
                width = 0.25
                
                precisions = [report[cn]['precision'] for cn in st.session_state.class_names if cn in report]
                recalls = [report[cn]['recall'] for cn in st.session_state.class_names if cn in report]
                f1s = [report[cn]['f1-score'] for cn in st.session_state.class_names if cn in report]
                
                ax.bar(x - width, precisions, width, label='Precision', color='#667eea')
                ax.bar(x, recalls, width, label='Recall', color='#f093fb')
                ax.bar(x + width, f1s, width, label='F1-Score', color='#2ecc71')
                
                ax.set_xlabel('Class', fontweight='bold')
                ax.set_ylabel('Score', fontweight='bold')
                ax.set_title('Per-Class Metrics Comparison', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(st.session_state.class_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col_vis2:
                fig, ax = plt.subplots(figsize=(10, 6))
                supports = [report[cn]['support'] for cn in st.session_state.class_names if cn in report]
                colors = plt.cm.viridis(np.linspace(0, 1, len(supports)))
                bars = ax.barh(st.session_state.class_names, supports, color=colors)
                ax.set_xlabel('Number of Samples', fontweight='bold')
                ax.set_title('Sample Distribution by Class', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, supports)):
                    ax.text(val + max(supports)*0.01, i, str(int(val)), 
                           va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    st.markdown("---")
    
    # Confusion Matrix and Correlation Heatmap
    st.markdown("## 🔍 Model Diagnostics")
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.markdown("### 📊 Confusion Matrix")
        st.markdown('<div class="info-box">Shows how well the model classifies each class. Diagonal values represent correct predictions.</div>', unsafe_allow_html=True)
        
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        # Use class names for labels
        class_labels = st.session_state.class_names if st.session_state.class_names is not None else [f"Class {i}" for i in range(len(np.unique(st.session_state.y_test)))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'},
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    annot_kws={'size': 11, 'weight': 'bold'})
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_diag2:
        st.markdown("### 🔥 Feature Correlation Heatmap")
        st.markdown('<div class="info-box">Shows relationships between features. High correlation (close to 1 or -1) indicates strong relationships.</div>', unsafe_allow_html=True)
        
        corr_matrix = st.session_state.X_train.corr()
        
        # Determine appropriate figure size based on number of features
        n_features = len(corr_matrix)
        fig_size = max(10, n_features * 0.8)
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Create heatmap with better readability
        sns.heatmap(corr_matrix, 
                    annot=True,  # Show values
                    fmt='.2f',   # Format to 2 decimal places
                    cmap='coolwarm', 
                    center=0, 
                    ax=ax,
                    square=True, 
                    linewidths=1.5,
                    linecolor='white',
                    cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                    annot_kws={'size': 9, 'weight': 'bold'},  # Larger, bold text
                    vmin=-1, vmax=1)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Feature Importance",
        "🔵 SHAP Analysis",
        "🟢 LIME Analysis",
        "⚖️ SHAP vs LIME"
    ])
    
    # Tab 1: Feature Importance
    with tab1:
        st.markdown("### 📊 Feature Importance Analysis")
        st.markdown('<div class="info-box">Displays which features have the most influence on model predictions globally across all samples.</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state.model, 'feature_importances_'):
            # Get feature importances
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
                bars = ax.barh(range(len(indices)), importances[indices], color=colors)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([st.session_state.feature_names[i] for i in indices])
                ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                ax.set_title('Feature Importance Rankings', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 🏆 Top 5 Features")
                importance_df = pd.DataFrame({
                    'Feature': [st.session_state.feature_names[i] for i in indices[:5]],
                    'Importance': [f"{importances[i]:.4f}" for i in indices[:5]],
                    'Rank': range(1, 6)
                })
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### 📖 Interpretation")
                st.markdown("""
                **How to read this:**
                - Higher values = more important
                - Top features drive predictions most
                - Bottom features have minimal impact
                """)
        else:
            st.info("ℹ️ Selected model doesn't support feature importance.")
    
    # Tab 2: SHAP Analysis
    with tab2:
        st.markdown("### 🔵 SHAP (SHapley Additive exPlanations)")
        st.markdown('<div class="info-box">SHAP uses game theory to calculate each feature\'s contribution to predictions. It answers: "How much did each feature push the prediction up or down?"</div>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Computing SHAP values..."):
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(st.session_state.model)
                shap_values = explainer.shap_values(st.session_state.X_test[:100])
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values_binary = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                else:
                    shap_values_binary = shap_values
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Summary Plot (Bee Swarm)")
                    st.markdown('<div class="explanation-box"><b>How to read:</b> Each dot is one sample. Color shows feature value (red=high, blue=low). Position shows impact on prediction (right=increases, left=decreases).</div>', unsafe_allow_html=True)
                    
                    # For multi-class, show plots for each class
                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        st.info(f"📊 Showing SHAP values for all {len(shap_values)} classes")
                        for class_idx in range(len(shap_values)):
                            class_label = st.session_state.class_names[class_idx] if st.session_state.class_names is not None else f"Class {class_idx}"
                            st.markdown(f"##### Class: {class_label}")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(
                                shap_values[class_idx], 
                                st.session_state.X_test[:100],
                                feature_names=st.session_state.feature_names,
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        # Binary classification or already selected class
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values_binary, 
                            st.session_state.X_test[:100],
                            feature_names=st.session_state.feature_names,
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    st.markdown("#### Bar Plot (Average Impact)")
                    st.markdown('<div class="explanation-box"><b>How to read:</b> Shows average absolute impact of each feature. Longer bars = more important features on average.</div>', unsafe_allow_html=True)
                    
                    # For multi-class, show bar plots for each class
                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        for class_idx in range(len(shap_values)):
                            class_label = st.session_state.class_names[class_idx] if st.session_state.class_names is not None else f"Class {class_idx}"
                            st.markdown(f"##### Class: {class_label}")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(
                                shap_values[class_idx],
                                st.session_state.X_test[:100],
                                feature_names=st.session_state.feature_names,
                                plot_type="bar",
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values_binary,
                            st.session_state.X_test[:100],
                            feature_names=st.session_state.feature_names,
                            plot_type="bar",
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Individual prediction explanation
                st.markdown("---")
                st.markdown("#### 🔍 Individual Sample Analysis")
                sample_idx = st.slider(
                    "Select test sample to analyze",
                    0,
                    min(99, len(st.session_state.X_test)-1),
                    0
                )
                
                # Show sample details
                st.markdown("##### 📋 Sample Details")
                sample_data = st.session_state.X_test_original.iloc[sample_idx]
                
                col_details1, col_details2 = st.columns(2)
                
                with col_details1:
                    st.markdown("**Original Values:**")
                    sample_df = pd.DataFrame({
                        'Feature': sample_data.index,
                        'Value': sample_data.values
                    })
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)
                
                with col_details2:
                    # Safely get actual class
                    if hasattr(st.session_state.y_test, 'iloc'):
                        actual_class_encoded = st.session_state.y_test.iloc[sample_idx]
                    else:
                        actual_class_encoded = st.session_state.y_test[sample_idx]
                    
                    # Convert to class name
                    if st.session_state.class_names is not None:
                        actual_class = st.session_state.class_names[actual_class_encoded]
                        predicted_class = st.session_state.class_names[y_pred[sample_idx]]
                    else:
                        actual_class = actual_class_encoded
                        predicted_class = y_pred[sample_idx]
                    
                    pred_proba = st.session_state.model.predict_proba(
                        st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                    )[0]
                    
                    st.markdown("**Prediction Info:**")
                    st.metric("Actual Class", actual_class)
                    st.metric("Predicted Class", predicted_class)
                    st.metric("Confidence", f"{max(pred_proba):.2%}")
                    
                    # Show all class probabilities
                    st.markdown("**All Class Probabilities:**")
                    for i, prob in enumerate(pred_proba):
                        class_label = st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}"
                        st.write(f"{class_label}: {prob:.2%}")
                
                st.markdown("##### 💧 SHAP Waterfall Plot")
                st.markdown('<div class="explanation-box"><b>How to read:</b> Start from base value (expected model output). Red arrows push prediction higher, blue arrows push lower. Final value is the actual prediction.</div>', unsafe_allow_html=True)
                
                # For multi-class, show waterfall for all classes
                if isinstance(shap_values, list) and len(shap_values) > 2:
                    st.info(f"📊 Showing waterfall plots for all {len(shap_values)} classes")
                    
                    # Create tabs for each class
                    class_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" 
                                         for i in range(len(shap_values))])
                    
                    for class_idx, class_tab in enumerate(class_tabs):
                        with class_tab:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            base_val = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values[class_idx][sample_idx],
                                    base_values=base_val,
                                    data=st.session_state.X_test.iloc[sample_idx].values,
                                    feature_names=st.session_state.feature_names
                                ),
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    # Binary classification
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) == 2 else (explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0])
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values_binary[sample_idx],
                            base_values=base_val,
                            data=st.session_state.X_test.iloc[sample_idx].values,
                            feature_names=st.session_state.feature_names
                        ),
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ SHAP computation error: {str(e)}")
    
    # Tab 3: LIME Analysis
    with tab3:
        st.markdown("### 🟢 LIME (Local Interpretable Model-agnostic Explanations)")
        st.markdown('<div class="info-box">LIME explains individual predictions by creating a simple, interpretable model around that specific prediction. It answers: "If I change this feature slightly, how does the prediction change?"</div>', unsafe_allow_html=True)
        
        sample_idx_lime = st.slider(
            "Select test sample to analyze",
            0,
            len(st.session_state.X_test)-1,
            0,
            key="lime_slider"
        )
        
        # Show sample details
        st.markdown("#### 📋 Selected Sample Details")
        sample_data_lime = st.session_state.X_test_original.iloc[sample_idx_lime]
        
        col_lime1, col_lime2, col_lime3 = st.columns(3)
        
        with col_lime1:
            st.markdown("**Feature Values:**")
            sample_df_lime = pd.DataFrame({
                'Feature': sample_data_lime.index,
                'Value': sample_data_lime.values
            })
            st.dataframe(sample_df_lime, use_container_width=True, hide_index=True)
        
        with col_lime2:
            # Safely get actual class
            if hasattr(st.session_state.y_test, 'iloc'):
                actual_class_lime_encoded = st.session_state.y_test.iloc[sample_idx_lime]
            else:
                actual_class_lime_encoded = st.session_state.y_test[sample_idx_lime]
            
            predicted_class_lime_encoded = y_pred[sample_idx_lime]
            
            # Convert to class names
            if st.session_state.class_names is not None:
                actual_class_lime = st.session_state.class_names[actual_class_lime_encoded]
                predicted_class_lime = st.session_state.class_names[predicted_class_lime_encoded]
            else:
                actual_class_lime = actual_class_lime_encoded
                predicted_class_lime = predicted_class_lime_encoded
            
            st.markdown("**Classification:**")
            st.metric("Actual Class", actual_class_lime)
            st.metric("Predicted Class", predicted_class_lime)
            st.metric("Match", "✅ Correct" if actual_class_lime_encoded == predicted_class_lime_encoded else "❌ Incorrect")
        
        with col_lime3:
            pred_proba_lime = st.session_state.model.predict_proba(
                st.session_state.X_test.iloc[sample_idx_lime:sample_idx_lime+1]
            )[0]
            
            st.markdown("**Confidence Scores:**")
            for i, prob in enumerate(pred_proba_lime):
                class_label = st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}"
                st.progress(float(prob), text=f"{class_label}: {prob:.2%}")
        
        st.markdown("---")
        
        with st.spinner("🔄 Generating LIME explanation..."):
            try:
                # Create LIME explainer
                explainer_lime = lime_tabular.LimeTabularExplainer(
                    st.session_state.X_train.values,
                    feature_names=st.session_state.feature_names,
                    class_names=st.session_state.class_names if st.session_state.class_names is not None else [f'Class {i}' for i in range(len(pred_proba_lime))],
                    mode='classification'
                )
                
                # Generate explanation
                exp = explainer_lime.explain_instance(
                    st.session_state.X_test.iloc[sample_idx_lime].values,
                    st.session_state.model.predict_proba,
                    num_features=len(st.session_state.feature_names)
                )
                
                st.markdown("#### 📊 LIME Feature Contributions")
                st.markdown('<div class="explanation-box"><b>How to read:</b> Green bars support the predicted class, orange bars oppose it. Longer bars = stronger influence. Values in conditions show the actual feature value for this sample.</div>', unsafe_allow_html=True)
                
                # Check if multi-class
                num_classes_lime = len(pred_proba_lime)
                
                if num_classes_lime > 2:
                    st.info(f"📊 Showing LIME explanations for all {num_classes_lime} classes")
                    
                    # Create tabs for each class
                    lime_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" 
                                        for i in range(num_classes_lime)])
                    
                    for class_idx, lime_tab in enumerate(lime_tabs):
                        with lime_tab:
                            # Generate explanation for this specific class
                            exp_class = explainer_lime.explain_instance(
                                st.session_state.X_test.iloc[sample_idx_lime].values,
                                st.session_state.model.predict_proba,
                                num_features=len(st.session_state.feature_names),
                                labels=[class_idx]
                            )
                            
                            fig = exp_class.as_pyplot_figure(label=class_idx)
                            fig.set_size_inches(12, 6)
                            class_label = st.session_state.class_names[class_idx] if st.session_state.class_names is not None else f"Class {class_idx}"
                            plt.title(f'LIME Feature Contributions for {class_label}', fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    # Binary classification
                    fig = exp.as_pyplot_figure()
                    fig.set_size_inches(12, 6)
                    plt.title('LIME Feature Contributions for Predicted Class', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Add LIME interpretation guide
                st.markdown("""
                <div class="explanation-box">
                <b>📖 How to Read LIME Chart:</b><br><br>
                <b>🟢 Green bars (positive values):</b> These features push the prediction TOWARDS the predicted class. Longer green bars = stronger support for the prediction.<br><br>
                <b>🔴 Red/Orange bars (negative values):</b> These features push the prediction AWAY from the predicted class. Longer bars = stronger opposition.<br><br>
                <b>Numbers on bars:</b> Show the strength of contribution (e.g., +0.15 means this feature increases probability by ~15%).<br><br>
                <b>Feature conditions:</b> The text shows the actual value of each feature for this specific sample (e.g., "age <= 25" means this person is 25 or younger).
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ LIME computation error: {str(e)}")
    
    # Tab 4: Comparison
    with tab4:
        st.markdown("### ⚖️ SHAP vs LIME Comparison")
        st.markdown('<div class="info-box">Compare how SHAP and LIME explain the same prediction. SHAP is globally consistent but slower; LIME is faster but can vary between runs.</div>', unsafe_allow_html=True)
        
        comparison_idx = st.slider(
            "Select sample for comparison",
            0,
            min(99, len(st.session_state.X_test)-1),
            0,
            key="comparison_slider"
        )
        
        # Show sample info
        st.markdown("#### 📋 Sample Information")
        sample_comp = st.session_state.X_test_original.iloc[comparison_idx]
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.markdown("**Features:**")
            comp_df = pd.DataFrame({
                'Feature': sample_comp.index,
                'Value': sample_comp.values
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        with col_comp2:
            st.markdown("**Prediction:**")
            # Safely get actual class
            if hasattr(st.session_state.y_test, 'iloc'):
                actual_comp_encoded = st.session_state.y_test.iloc[comparison_idx]
            else:
                actual_comp_encoded = st.session_state.y_test[comparison_idx]
            
            predicted_comp_encoded = y_pred[comparison_idx]
            
            # Convert to class names
            if st.session_state.class_names is not None:
                actual_comp = st.session_state.class_names[actual_comp_encoded]
                predicted_comp = st.session_state.class_names[predicted_comp_encoded]
            else:
                actual_comp = actual_comp_encoded
                predicted_comp = predicted_comp_encoded
            
            st.metric("Actual", actual_comp)
            st.metric("Predicted", predicted_comp)
        
        with col_comp3:
            pred_proba_comp = st.session_state.model.predict_proba(
                st.session_state.X_test.iloc[comparison_idx:comparison_idx+1]
            )[0]
            st.markdown("**Confidence:**")
            st.metric("Probability", f"{max(pred_proba_comp):.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 SHAP Explanation")
            st.markdown('<div class="explanation-box"><b>Method:</b> Uses game theory to fairly distribute prediction credit among features. Guarantees consistency - same feature values always get same credit.</div>', unsafe_allow_html=True)
            try:
                explainer_comp = shap.TreeExplainer(st.session_state.model)
                shap_values_comp = explainer_comp.shap_values(st.session_state.X_test[:100])
                
                # Determine number of classes
                num_classes_comp = len(pred_proba_comp)
                
                if num_classes_comp > 2 and isinstance(shap_values_comp, list):
                    st.info(f"📊 Showing SHAP waterfall for all {num_classes_comp} classes")
                    
                    # Create tabs for each class
                    shap_comp_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" 
                                             for i in range(num_classes_comp)])
                    
                    for class_idx, shap_tab in enumerate(shap_comp_tabs):
                        with shap_tab:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            base_val = explainer_comp.expected_value[class_idx] if isinstance(explainer_comp.expected_value, np.ndarray) else explainer_comp.expected_value
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values_comp[class_idx][comparison_idx],
                                    base_values=base_val,
                                    data=st.session_state.X_test.iloc[comparison_idx].values,
                                    feature_names=st.session_state.feature_names
                                ),
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    # Binary classification
                    if isinstance(shap_values_comp, list):
                        shap_values_comp = shap_values_comp[1] if len(shap_values_comp) == 2 else shap_values_comp[0]
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values_comp[comparison_idx],
                            base_values=explainer_comp.expected_value if not isinstance(explainer_comp.expected_value, np.ndarray) else explainer_comp.expected_value[0],
                            data=st.session_state.X_test.iloc[comparison_idx].values,
                            feature_names=st.session_state.feature_names
                        ),
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Add SHAP interpretation guide
                st.markdown("""
                <div class="explanation-box">
                <b>📖 How to Read SHAP Waterfall Chart:</b><br><br>
                <b>Starting Point (E[f(X)]):</b> This is the baseline - the average prediction across all training data.<br><br>
                <b>🔴 Red arrows (pointing right):</b> These features INCREASE the prediction. They push the value higher than the baseline.<br><br>
                <b>🔵 Blue arrows (pointing left):</b> These features DECREASE the prediction. They push the value lower than the baseline.<br><br>
                <b>Arrow length:</b> Longer arrows = stronger impact on the prediction.<br><br>
                <b>Final value f(x):</b> The actual prediction after all features have contributed. This is where all the arrows lead to.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"SHAP error: {str(e)}")
        
        with col2:
            st.markdown("#### 🟢 LIME Explanation")
            st.markdown('<div class="explanation-box"><b>Method:</b> Perturbs data around the prediction point and fits a simple linear model locally. Fast but can give different results on repeated runs.</div>', unsafe_allow_html=True)
            try:
                explainer_lime_comp = lime_tabular.LimeTabularExplainer(
                    st.session_state.X_train.values,
                    feature_names=st.session_state.feature_names,
                    class_names=st.session_state.class_names if st.session_state.class_names is not None else [f'Class {i}' for i in range(len(pred_proba_comp))],
                    mode='classification'
                )
                
                num_classes_lime_comp = len(pred_proba_comp)
                
                if num_classes_lime_comp > 2:
                    st.info(f"📊 Showing LIME explanations for all {num_classes_lime_comp} classes")
                    
                    # Create tabs for each class
                    lime_comp_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" 
                                             for i in range(num_classes_lime_comp)])
                    
                    for class_idx, lime_tab in enumerate(lime_comp_tabs):
                        with lime_tab:
                            exp_comp = explainer_lime_comp.explain_instance(
                                st.session_state.X_test.iloc[comparison_idx].values,
                                st.session_state.model.predict_proba,
                                num_features=len(st.session_state.feature_names),
                                labels=[class_idx]
                            )
                            
                            fig = exp_comp.as_pyplot_figure(label=class_idx)
                            fig.set_size_inches(8, 5)
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    # Binary classification
                    exp_comp = explainer_lime_comp.explain_instance(
                        st.session_state.X_test.iloc[comparison_idx].values,
                        st.session_state.model.predict_proba,
                        num_features=len(st.session_state.feature_names)
                    )
                    
                    fig = exp_comp.as_pyplot_figure()
                    fig.set_size_inches(8, 5)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Add LIME interpretation guide
                st.markdown("""
                <div class="explanation-box">
                <b>📖 How to Read LIME Chart:</b><br><br>
                <b>🟢 Green bars (positive values):</b> These features push the prediction TOWARDS the predicted class. Longer green bars = stronger support for the prediction.<br><br>
                <b>🔴 Red/Orange bars (negative values):</b> These features push the prediction AWAY from the predicted class. Longer bars = stronger opposition.<br><br>
                <b>Numbers on bars:</b> Show the strength of contribution (e.g., +0.15 means this feature increases probability by ~15%).<br><br>
                <b>Feature conditions:</b> The text shows the actual value of each feature for this specific sample (e.g., "age <= 25" means this person is 25 or younger).
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"LIME error: {str(e)}")
        
        # Comparison summary
        st.markdown("---")
        st.markdown("### 📊 Method Comparison Summary")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("""
            **🔵 SHAP Characteristics:**
            - **Based on:** Game theory (Shapley values)
            - **Scope:** Global and local explanations
            - **Consistency:** Mathematically guaranteed
            - **Speed:** Slower computation
            - **Best for:** Tree-based models
            - **Stability:** Same input = same output
            - **Interpretation:** Additive feature contributions
            """)
        
        with comp_col2:
            st.markdown("""
            **🟢 LIME Characteristics:**
            - **Based on:** Local perturbations
            - **Scope:** Local explanations only
            - **Consistency:** Not guaranteed
            - **Speed:** Faster computation
            - **Best for:** Any model (model-agnostic)
            - **Stability:** May vary between runs
            - **Interpretation:** Linear approximation
            """)
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <b>💡 When to use which?</b><br>
        <b>Use SHAP when:</b> You need consistent, theoretically sound explanations and work with tree-based models.<br>
        <b>Use LIME when:</b> You need quick explanations, work with any model type, or need simple linear interpretations.
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 Welcome to Explainable AI Dashboard!</h2>
        <p style='font-size: 1.2rem; color: #666; margin-top: 1rem;'>
            Get started by uploading a CSV file from the sidebar and training your model.
        </p>
        <br>
        <p style='color: #888;'>
            This dashboard helps you understand <b>how</b> and <b>why</b> your machine learning model makes predictions.<br>
            Explore feature importance, SHAP values, and LIME explanations to gain deep insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;'>
            <h2>🎯</h2>
            <h3>Feature Importance</h3>
            <p>Discover which features matter most in your model's decisions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;'>
            <h2>🔍</h2>
            <h3>SHAP & LIME</h3>
            <p>Compare two powerful explainability techniques side-by-side</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;'>
            <h2>📊</h2>
            <h3>Visual Analysis</h3>
            <p>Interactive charts and detailed performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick guide
    st.markdown("### 🚀 Quick Start Guide")
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.markdown("""
        **Step 1: Upload Data** 📁
        - Click "Browse files" in the sidebar
        - Select your CSV file
        - Preview your data
        
        **Step 2: Configure Model** ⚙️
        - Select target variable
        - Choose features to include
        - Pick a machine learning algorithm
        """)
    
    with guide_col2:
        st.markdown("""
        **Step 3: Train & Analyze** 🎯
        - Adjust hyperparameters if needed
        - Click "Train Model"
        - Explore results across 4 analysis tabs
        
        **Step 4: Interpret Results** 💡
        - Review model performance metrics
        - Analyze feature importance
        - Compare SHAP and LIME explanations
        """)
    
    st.markdown("---")
    
    st.info("💡 **Tip:** This dashboard works best with classification datasets. Make sure your target variable has discrete classes.")