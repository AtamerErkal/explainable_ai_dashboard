import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            f1_score, recall_score, precision_score, roc_curve, auc, roc_auc_score)
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

# Custom CSS styling with Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --secondary-gradient: linear-gradient(135deg, #3b82f6 0%, #2dd4bf 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Domain Responsive Styles */
    .domain-Defence { border-top: 5px solid #ef4444; }
    .domain-Healthcare { border-top: 5px solid #10b981; }
    .domain-Finance { border-top: 5px solid #f59e0b; }
    .domain-General { border-top: 5px solid #6366f1; }

    .main-header {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.05em;
        background: linear-gradient(to right, #1e293b, #475569);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Premium Card Design */
    .premium-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.25rem;
        transition: transform 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
    }

    .description-box {
        background: var(--primary-gradient);
        padding: 1.25rem;
        border-radius: 16px;
        color: white !important;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.3);
    }
    
    .description-box h3, .description-box p, .description-box strong {
        color: white !important;
    }

    /* Metric & Summary Boxes */
    .metric-box, .data-summary-box {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .data-summary-box h2, .data-summary-box h3, .data-summary-box p {
        color: #1e293b !important;
    }

    .metric-box:hover, .data-summary-box:hover {
        border-color: #6366f1;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2);
    }

    .metric-title {
        color: #475569;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Information & Explanation Boxes */
    .info-box {
        background: #f8fafc;
        padding: 0.6rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin: 0.5rem 0;
        color: #1e293b !important;
        font-size: 0.8rem;
    }
    
    .explanation-box {
        background: #f0fdf4;
        padding: 0.75rem;
        border-radius: 12px;
        border-left: 5px solid #22c55e;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        color: #14532d !important;
        line-height: 1.5;
    }

    /* Workflow Step Cards */
    .workflow-container {
        display: flex;
        gap: 15px;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    .workflow-step {
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        flex: 1;
        min-width: 200px;
        border: 1px solid #e2e8f0;
        position: relative;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .workflow-step:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #6366f1;
    }

    .step-number {
        width: 32px;
        height: 32px;
        background: var(--primary-gradient);
        color: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    .step-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }

    .step-desc {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Feature Highlight Glows */
    .glow-purple { border-top: 4px solid #a855f7; }
    .glow-blue { border-top: 4px solid #3b82f6; }
    .glow-teal { border-top: 4px solid #2dd4bf; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button[kind="primary"] {
        background: var(--primary-gradient);
        border: none;
        color: white;
    }

    /* Modern Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f1f5f9;
        padding: 10px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px !important;
        border-radius: 12px;
        background-color: transparent;
        border: 1px solid transparent !important;
        color: #64748b;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(99, 102, 241, 0.05);
        color: #6366f1;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #6366f1 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        transform: scale(1.02);
    }

    /* Tooltip styles */
    .metric-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
    }
    
    .info-icon {
        cursor: help;
        font-size: 0.8rem;
        background: #e2e8f0;
        color: #64748b;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 4px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade {
        animation: fadeIn 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-header">🔍 AI Explainability Governance Dashboard</div>', unsafe_allow_html=True)

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
if 'domain' not in st.session_state:
    st.session_state.domain = "General"

# Sidebar configuration
with st.sidebar:
    st.markdown(f'<div class="sidebar-header">🛡️ Protocol Control</div>', unsafe_allow_html=True)
    
    with st.expander("🌐 Operational Domain", expanded=True):
        st.session_state.domain = st.selectbox("Industry Context", ["General", "Healthcare", "Defence", "Finance"])
    
    st.markdown("---")
    
    with st.expander("🧪 Intelligence Intake", expanded=True):
        # Sample data loader
        st.markdown("**Domain Samples**")
        if st.button("Load Domain Sample Data", use_container_width=True):
            sample_files = {
                "Healthcare": "sample_data/healthcare_data.csv",
                "Defence": "sample_data/defence_data.csv",
                "Finance": "sample_data/finance_data.csv",
                "General": "sample_data/general_data.csv"
            }
            file_path = sample_files.get(st.session_state.domain)
            if file_path and os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.session_state.original_df = df.copy()
                st.success(f"✅ Context: {st.session_state.domain}")
            else:
                st.warning("Sample not found.")

        st.markdown("**Manual Upload**")
        uploaded_file = st.file_uploader("Select CSV", type=['csv'])
        if uploaded_file:
            st.session_state.original_df = pd.read_csv(uploaded_file)

    st.markdown("---")
    
    if st.session_state.original_df is not None:
        df = st.session_state.original_df
        
        with st.expander("🤖 AI Reactor Engine", expanded=True):
            target_col = st.selectbox("Target Variable", df.columns, index=len(df.columns)-1)
            available_features = [col for col in df.columns if col != target_col]
            feature_cols = st.multiselect("Feature Selection", available_features, default=available_features[:min(8, len(available_features))])
            
            if len(feature_cols) > 0:
                model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
                if model_choice == "Random Forest":
                    n_estimators = st.slider(
                        "n_estimators", 10, 200, 100, 10, key="rf_n",
                        help="**What:** Number of trees in the forest.\n\n**Effect:**\n- **Increase (+):** Better stability and accuracy, but slower training.\n- **Decrease (-):** Faster, but higher risk of variance.\n\n**Pro/Con:** General rule: Higher is better, but with diminishing returns."
                    )
                    max_depth = st.slider(
                        "max_depth", 3, 20, 10, key="rf_d",
                        help="**What:** Maximum depth of each tree.\n\n**Effect:**\n- **Increase (+):** Learns complex patterns, but high risk of **OVERFITTING**.\n- **Decrease (-):** Simpler model, prevents overfitting but might **UNDERFIT**.\n\n**Pro/Con:** Crucial for balancing model complexity."
                    )
                elif model_choice == "Gradient Boosting":
                    n_estimators = st.slider(
                        "n_estimators", 10, 200, 100, 10, key="gb_n",
                        help="**What:** Number of sequential boosting stages.\n\n**Effect:**\n- **Increase (+):** More chances to correct errors, but high risk of **OVERFITTING**.\n- **Decrease (-):** Faster, safer against overfitting but might miss patterns.\n\n**Pro/Con:** Unlike RF, too many trees will definitely overfit here."
                    )
                    learning_rate = st.slider(
                        "learning_rate", 0.01, 0.3, 0.1, 0.01, key="gb_l",
                        help="**What:** Shrinkage applied to each new tree's contribution.\n\n**Effect:**\n- **Increase (+):** Faster learning, but might 'jump over' the best solution.\n- **Decrease (-):** More precise learning, but requires MORE n_estimators.\n\n**Pro/Con:** Low rate + High n_estimators = Best Performance (but slow)."
                    )
                else:
                    max_iter = st.slider(
                        "max_iter", 100, 1000, 200, 100, key="lr_i",
                        help="**What:** Max iterations for the solver to converge.\n\n**Effect:**\n- **Increase (+):** Ensures model finds the best fit on complex data.\n- **Decrease (-):** Faster execution, but risks 'No Convergence' (sub-optimal model)."
                    )
                
                test_size = st.slider("Split (%)", 10, 50, 20, 5) / 100
                random_state = 42

        # Train button
        if st.button("🏁 EXECUTE TRAINING", use_container_width=True, type="primary"):
            with st.spinner("🔄 INITIALIZING REACTOR..."):
                try:
                    # Prepare data
                    X = df[feature_cols].copy()
                    y = df[target_col].copy()
                    
                    # Store original X for later display
                    X_original = X.copy()
                    
                    # Aggressive numeric cleansing (Fix for hidden spaces/symbols in Churn data)
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            # Strip common junk characters that break numeric conversion
                            cleaned_col = X[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
                            # Replace empty strings with NaN
                            cleaned_col = cleaned_col.replace('', np.nan)
                            
                            # Attempt final conversion
                            converted = pd.to_numeric(cleaned_col, errors='coerce')
                            if not converted.isna().all(): # If at least some values are numbers
                                X[col] = converted
                    
                    # Handle missing values after conversion (Fill NaNs with mean)
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
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # PDF Export block
            st.markdown("---")
            st.markdown("### 📥 Export Audit Report")
            if st.button("Generate PDF Report", use_container_width=True):
                if st.session_state.model is not None:
                    with st.spinner("Generating PDF..."):
                        try:
                            from fpdf import FPDF
                            import datetime
                            import matplotlib.pyplot as plt
                            import shap
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                            import seaborn as sns
                            import numpy as np
                            import tempfile
                            
                            class PDFReport(FPDF):
                                def header(self):
                                    self.set_fill_color(31, 41, 55)
                                    self.rect(0, 0, 210, 25, 'F')
                                    self.set_y(8)
                                    self.set_font("Helvetica", "B", 16)
                                    self.set_text_color(255, 255, 255)
                                    domain_str = st.session_state.domain.upper() if st.session_state.domain else "UNKNOWN"
                                    self.cell(0, 10, f"{domain_str} - AI EXPLAINABILITY AUDIT", align='C', ln=True)
                                    self.set_y(25)
                                    self.set_text_color(0, 0, 0)
                                    
                                def footer(self):
                                    self.set_y(-15)
                                    self.set_font("Helvetica", "I", 8)
                                    self.set_text_color(128, 128, 128)
                                    self.cell(0, 10, f"Page {self.page_no()}/{{nb}} - Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", align='C')
                            
                            pdf = PDFReport()
                            pdf.alias_nb_pages()
                            pdf.add_page()
                            
                            # Domain Interpretations
                            pdf.set_y(30)
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.set_text_color(31, 41, 55)
                            pdf.cell(0, 10, "Executive Summary & Domain Context", ln=True)
                            
                            pdf.set_font("Helvetica", "", 10)
                            pdf.set_text_color(50, 50, 50)
                            if st.session_state.domain == "Healthcare":
                                exec_summary = "This report evaluates medical predictive models. In the healthcare domain, minimizing False Negatives (maximizing Recall) is critical. A high recall ensures that critical patient anomalies are not missed, which could delay life-saving interventions."
                            elif st.session_state.domain == "Defence":
                                exec_summary = "This report evaluates defence and security predictive systems. In defence implementations, robust precision and minimizing False Positives are paramount to avoid unintended critical actions and ensure absolute operational certainty."
                            else:
                                exec_summary = "This report summarizes the interpretability and performance metrics of the underlying machine learning model, supporting trust and accountability in automated decisions."
                            pdf.multi_cell(0, 6, exec_summary)
                            pdf.ln(5)
                            
                            # Model info
                            pdf.set_text_color(0, 0, 0)
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.cell(0, 8, "Model Configuration", ln=True)
                            pdf.set_font("Helvetica", "", 10)
                            pdf.cell(0, 6, f"Algorithm: {st.session_state.get('model_name', 'N/A')}", ln=True)
                            pdf.cell(0, 6, f"Target Variable: {st.session_state.get('target_name', 'N/A')}", ln=True)
                            pdf.cell(0, 6, f"Dataset Columns: {len(st.session_state.feature_names) if st.session_state.feature_names else 0} features", ln=True)
                            pdf.ln(4)
                            
                            # Performance metrics
                            y_pred = st.session_state.model.predict(st.session_state.X_test)
                            acc = accuracy_score(st.session_state.y_test, y_pred)
                            prec = precision_score(st.session_state.y_test, y_pred, average='macro', zero_division=0)
                            rec = recall_score(st.session_state.y_test, y_pred, average='macro', zero_division=0)
                            f1 = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                            
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.cell(0, 8, "Performance Metrics", ln=True)
                            
                            pdf.set_font("Helvetica", "B", 11)
                            pdf.set_fill_color(240, 240, 240)
                            pdf.cell(95, 8, "Metric", border=1, fill=True)
                            pdf.cell(95, 8, "Score", border=1, ln=True, fill=True)
                            
                            metrics = [("Accuracy", f"{acc:.4f} ({acc:.2%})"),
                                       ("Precision", f"{prec:.4f} ({prec:.2%})"),
                                       ("Recall", f"{rec:.4f} ({rec:.2%})"),
                                       ("F1 Score", f"{f1:.4f} ({f1:.2%})")]
                                       
                            for m_name, m_score in metrics:
                                if st.session_state.domain == "Healthcare" and m_name == "Recall":
                                    pdf.set_font("Helvetica", "B", 10)
                                    pdf.set_text_color(180, 0, 0)
                                elif st.session_state.domain == "Defence" and m_name == "Precision":
                                    pdf.set_font("Helvetica", "B", 10)
                                    pdf.set_text_color(180, 0, 0)
                                else:
                                    pdf.set_font("Helvetica", "", 10)
                                    pdf.set_text_color(50, 50, 50)
                                    
                                pdf.cell(95, 8, m_name, border=1)
                                pdf.cell(95, 8, m_score, border=1, ln=True)
                                
                            pdf.set_text_color(0, 0, 0)
                            pdf.ln(2)
                            
                            # Educational context in PDF
                            pdf.set_font("Helvetica", "I", 9)
                            pdf.set_text_color(100, 100, 100)
                            
                            domain_context = ""
                            if st.session_state.domain == "Healthcare":
                                domain_context = "In Healthcare, Recall is the most critical metric as it represents the model's ability to catch every positive case (e.g., Disease). A false negative (missing a patient) is much costlier than a false positive."
                            elif st.session_state.domain == "Defence":
                                domain_context = "In Defence, Precision is often prioritized to avoid false engagement of targets. High precision ensures that when a target is classified as hostile, it is almost certainly a valid target."
                            else:
                                domain_context = "The Balance between Precision and Recall should be tuned based on the 'Cost of Error' in the specific operational environment."
                                
                            pdf.multi_cell(0, 5, f"Interpretation Note: {domain_context} The current F1-Score of {f1:.2%} shows the overall harmonic robustness of this intelligence model.")
                            pdf.set_text_color(0, 0, 0)
                            pdf.ln(5)
                            
                            # ROC Curve in PDF
                            try:
                                y_prob = st.session_state.model.predict_proba(st.session_state.X_test)
                                if len(st.session_state.class_names) == 2:
                                    fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob[:, 1])
                                    roc_auc = auc(fpr, tpr)
                                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                                    ax_roc.plot(fpr, tpr, color='#6366f1', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                                    ax_roc.plot([0, 1], [0, 1], color='grey', linestyle='--')
                                    ax_roc.set_title('ROC Curve for Binary Classification')
                                    ax_roc.legend(loc="lower right")
                                    plt.tight_layout()
                                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_roc:
                                        fig_roc.savefig(tmp_roc.name, format="png")
                                        pdf.image(tmp_roc.name, w=100, x=55)
                                    plt.close(fig_roc)
                                    pdf.ln(4)
                                    pdf.set_font("Helvetica", "B", 10)
                                    pdf.cell(0, 6, f"AUC Score: {roc_auc:.3f} - Represents the probability that the model ranks a random positive above a random negative.", ln=True)
                            except:
                                pass
                            
                            # Confusion Matrix
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.cell(0, 10, "Diagnostic Visualizations: Confusion Matrix", ln=True)
                            cm = confusion_matrix(st.session_state.y_test, y_pred)
                            
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                            class_labels = st.session_state.class_names if st.session_state.class_names is not None else [f"C{i}" for i in range(len(np.unique(st.session_state.y_test)))]
                            safe_classes = [str(c).replace('$', '\\$') for c in class_labels]
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=safe_classes, yticklabels=safe_classes)
                            ax_cm.set_ylabel('True Label')
                            ax_cm.set_xlabel('Predicted Label')
                            plt.tight_layout()
                            
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_cm:
                                fig_cm.savefig(tmp_cm.name, format="png", bbox_inches='tight')
                                pdf.image(tmp_cm.name, w=110, x=50)
                            plt.close(fig_cm)
                            
                            # Next page for SHAP and Feature Importance
                            pdf.add_page()
                            pdf.set_y(30)
                            
                            # Feature Importance
                            if hasattr(st.session_state.model, 'feature_importances_'):
                                pdf.set_font("Helvetica", "B", 14)
                                pdf.cell(0, 10, "Feature Importance Rankings", ln=True)
                                importances = st.session_state.model.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                                ax_fi.barh(range(min(10, len(indices))), importances[indices][:10], color='cornflowerblue')
                                ax_fi.set_yticks(range(min(10, len(indices))))
                                safe_features = [str(st.session_state.feature_names[i]).replace('$', '\\$') for i in indices[:10]]
                                ax_fi.set_yticklabels(safe_features)
                                ax_fi.invert_yaxis()
                                ax_fi.set_title('Top Driving Features for the Algorithm')
                                plt.tight_layout()
                                
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fi:
                                    fig_fi.savefig(tmp_fi.name, format="png", bbox_inches='tight')
                                    pdf.image(tmp_fi.name, w=150, x=30)
                                plt.close(fig_fi)
                                pdf.ln(5)
                                
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.cell(0, 10, "Global Feature Impact (SHAP Summary)", ln=True)
                            
                            if "LogisticRegression" in str(type(st.session_state.model)):
                                explainer = shap.LinearExplainer(st.session_state.model, st.session_state.X_train)
                            else:
                                explainer = shap.TreeExplainer(st.session_state.model)
                                
                            shap_values_raw = explainer.shap_values(st.session_state.X_test[:100])
                            if getattr(shap_values_raw, 'shape', None) and len(shap_values_raw.shape) == 3:
                                shap_values_list = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
                            elif isinstance(shap_values_raw, list):
                                shap_values_list = shap_values_raw
                            else:
                                shap_values_list = [shap_values_raw]
                                
                            shap_values_binary = shap_values_list[1] if len(shap_values_list) == 2 else shap_values_list[0]
                            
                            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                            
                            safe_feature_names = [str(f).replace('$', '\\$') for f in st.session_state.feature_names]
                            X_plot_safe = st.session_state.X_test[:100].copy()
                            X_plot_safe.columns = safe_feature_names
                            
                            shap.summary_plot(shap_values_binary, X_plot_safe, feature_names=safe_feature_names, show=False)
                            plt.tight_layout()
                            
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_shap:
                                fig_shap.savefig(tmp_shap.name, format="png", bbox_inches='tight')
                                pdf.image(tmp_shap.name, w=160, x=25)
                            plt.close(fig_shap)
                            pdf.ln(5)
                            
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.cell(0, 8, "LIME & Counterfactual Context", ln=True)
                            pdf.set_font("Helvetica", "", 10)
                            pdf.multi_cell(0, 6, "While SHAP demonstrates global contribution patterns, Local Interpretable Model-agnostic Explanations (LIME) and Counterfactual Generation (DiCE) are used extensively within the platform for precise single-instance troubleshooting. This ensures continuous verification against non-compliant AI drift scenarios.")
                            
                            st.session_state['pdf_bytes'] = bytes(pdf.output())
                            
                        except ImportError as e:
                            st.error(f"Missing libraries: {str(e)}")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.warning("Please train the model first.")
            
            if 'pdf_bytes' in st.session_state:
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state['pdf_bytes'],
                    file_name=f"XAI_Audit_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🛡️ System Status</div>', unsafe_allow_html=True)
    status_color = "#22c55e" if st.session_state.model else "#64748b"
    status_text = "Operational" if st.session_state.model else "Idle"
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {status_color};"></div>
            <div style="font-weight: 600; color: #1e293b;">{status_text}</div>
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
            Domain: {st.session_state.domain}<br>
            Model: {st.session_state.get('model_name', 'None')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content area
if st.session_state.model is not None:
    
    # Data Summary Section
    st.markdown("## 📊 Dataset Summary")
    
    if st.session_state.original_df is not None:
        df_summary = st.session_state.original_df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="data-summary-box animate-fade">
                <div class="metric-title">📋 Total Rows</div>
                <div class="metric-value">{df_summary.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="data-summary-box animate-fade">
                <div class="metric-title">📊 Total Columns</div>
                <div class="metric-value">{df_summary.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = df_summary.select_dtypes(include=[np.number]).shape[1]
            st.markdown(f"""
            <div class="data-summary-box animate-fade">
                <div class="metric-title">🔢 Numeric Features</div>
                <div class="metric-value">{numeric_cols}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_cols = df_summary.select_dtypes(include=['object']).shape[1]
            st.markdown(f"""
            <div class="data-summary-box animate-fade">
                <div class="metric-title">📝 Categorical Features</div>
                <div class="metric-value">{categorical_cols}</div>
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
        <div class="metric-box animate-fade">
            <div class="metric-title">🎯 Accuracy <span class="info-icon" title="Overall correctness: Ratio of correct predictions to total cases. Best for balanced datasets.">i</span></div>
            <div class="metric-value">{accuracy:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box animate-fade">
            <div class="metric-title">🎪 Precision <span class="info-icon" title="Quality: Out of all positive predictions, how many were actually correct? Important for avoiding false alarms.">i</span></div>
            <div class="metric-value">{precision:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box animate-fade">
            <div class="metric-title">🔄 Recall <span class="info-icon" title="Quantity: Out of all actual positive cases, how many did we find? Critical for health and safety.">i</span></div>
            <div class="metric-value">{recall:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box animate-fade">
            <div class="metric-title">⚖️ F1 Score <span class="info-icon" title="Balanced Harmonic Mean of Precision and Recall. Best overall metric for imbalanced data.">i</span></div>
            <div class="metric-value">{f1:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-box animate-fade">
            <div class="metric-title">📊 Features <span class="info-icon" title="Total number of input variables used to train the current model.">i</span></div>
            <div class="metric-value">{len(st.session_state.feature_names)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic Audit Verdict
    ver_col = "#22c55e" if accuracy > 0.8 else "#f59e0b" if accuracy > 0.7 else "#ef4444"
    st.markdown(f"""
    <div class="explanation-box" style="border-left-color: {ver_col}; background-color: {ver_col}10;">
        <strong>📋 Strategic Audit Verdict:</strong> 
        Model operational at <b>{accuracy:.1%} accuracy</b>. 
        {"Deployment status: VERIFIED. Governance protocols met." if accuracy > 0.8 else "Deployment status: PROVISIONAL. Secondary audit recommended." if accuracy > 0.7 else "Deployment status: CRITICAL. Does not meet safety thresholds."}
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
                fig, ax = plt.subplots(figsize=(5, 3))
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
                fig, ax = plt.subplots(figsize=(5, 3))
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
    
    col_diag1, col_diag2, col_diag3 = st.columns(3)
    
    with col_diag1:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 📊 Confusion Matrix")
        st.markdown("""
        <div class="info-box">
            <b>Truth Table:</b> Shows actual vs predicted counts. 
            Diagonal boxes (Top-Left, Bottom-Right) are <b>Correct</b> decisions. 
            Off-diagonal boxes represent <b>Type I / Type II errors</b> (Mistakes).
        </div>
        """, unsafe_allow_html=True)
        
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        class_labels = st.session_state.class_names if st.session_state.class_names is not None else [f"Class {i}" for i in range(len(np.unique(st.session_state.y_test)))]
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_diag2:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 📈 Separation Power (ROC)")
        st.markdown("""
        <div class="info-box">
            <b>Model Precision:</b> Visualizes the ability to distinguish between classes. 
            The higher the curve (Area Under Curve - <b>AUC</b>), the better. 
            <b>AUC > 0.8:</b> Strong separation. <b>0.5:</b> Random guessing.
        </div>
        """, unsafe_allow_html=True)
        try:
            y_prob = st.session_state.model.predict_proba(st.session_state.X_test)
            if len(st.session_state.class_names) == 2:
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, color='#6366f1', lw=2, label=f'AUC={roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='#cbd5e1', lw=1, linestyle='--')
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_roc)
            else:
                macro_auc = roc_auc_score(st.session_state.y_test, y_prob, multi_class='ovr', average='macro')
                st.metric("Macro AUC Score", f"{macro_auc:.3f}")
                st.info("Visual ROC support for Multi-class is in the detailed metrics above.")
        except:
            st.warning("ROC Visualization Unavailable for this model.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_diag3:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 🔥 Multi-Feature Correlation")
        st.markdown("""
        <div class="info-box">
            <b>Feature Relationships:</b> Measures linear dependency. 
            Values near <b>1</b> (Red) mean strong positive links. 
            Near <b>-1</b> (Blue) mean inverse links. 
            <i>Example: In churn data, Tenure and Contract type often correlate highly.</i>
        </div>
        """, unsafe_allow_html=True)
        corr_matrix = st.session_state.X_train.corr()
        fig_size = max(5, len(corr_matrix) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar=False, annot_kws={"size": 8})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Feature Importance",
        "🔵 SHAP Analysis",
        "🟢 LIME Analysis",
        "⚖️ SHAP vs LIME",
        "🔀 Counterfactual",
        "⚖️ Bias Audit"
    ])
    
    # Tab 1: Feature Importance
    with tab1:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 🎯 Global Feature Influence")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Identifies which variables the model prioritizes across all decisions. 
            The longer the bar, the more 'loyal' the model is to that specific feature when making a choice.
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state.model, 'feature_importances_'):
            # Get feature importances
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(8, 5))
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
                
                st.markdown("""
                <div class="explanation-box">
                    <strong>🔍 Interpretation Protocol:</strong>
                    <ul style='margin-bottom: 0;'>
                        <li>Higher values indicate a stronger causal link to the prediction.</li>
                        <li>Top features should align with domain expertise (e.g., Clinical logic).</li>
                        <li>Unexpectedly high importance may indicate data leakage or bias.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True) # End premium-card
        else:
            st.info("ℹ️ Selected model doesn't support feature importance.")
    
    # Tab 2: SHAP Analysis
    with tab2:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 🔵 SHAP Value Decomposition")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Mathematical breakdown of "Why this specific result?". 
            Red bars push the outcome <b>POSITIVE</b>, while blue bars pull it <b>NEGATIVE</b>. 
            <i>Powered by Game Theory for absolute consistency.</i>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Computing SHAP values..."):
            try:
                # Create appropriate SHAP explainer
                if "LogisticRegression" in str(type(st.session_state.model)):
                    explainer = shap.LinearExplainer(st.session_state.model, st.session_state.X_train)
                else:
                    explainer = shap.TreeExplainer(st.session_state.model)
                
                shap_values_raw = explainer.shap_values(st.session_state.X_test[:100])
                
                if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
                    shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
                else:
                    shap_values = shap_values_raw
                
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
                            fig, ax = plt.subplots(figsize=(5, 3))
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
                        fig, ax = plt.subplots(figsize=(5, 3))
                        shap.summary_plot(
                            shap_values_binary, 
                            st.session_state.X_test_original[:100],
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
                            fig, ax = plt.subplots(figsize=(5, 3))
                            shap.summary_plot(
                                shap_values[class_idx],
                                st.session_state.X_test_original[:100],
                                feature_names=st.session_state.feature_names,
                                plot_type="bar",
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        shap.summary_plot(
                            shap_values_binary,
                            st.session_state.X_test[:100],
                            feature_names=st.session_state.feature_names,
                            plot_type="bar",
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                
                st.markdown("---")
                st.markdown("#### 📉 Dependence Plot")
                st.markdown('<div class="explanation-box"><b>How to read:</b> Shows how a single feature affects the prediction. Color represents a secondary feature, automatically selected to highlight potential interaction effects.</div>', unsafe_allow_html=True)
                
                dep_feat = st.selectbox("Select feature for Dependence Plot", st.session_state.feature_names, key="dep_feat_select")
                
                if isinstance(shap_values, list) and len(shap_values) > 2:
                    st.info(f"📊 Showing Dependence plots for all {len(shap_values)} classes")
                    dep_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" for i in range(len(shap_values))])
                    
                    for class_idx, dep_tab in enumerate(dep_tabs):
                        with dep_tab:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            shap.dependence_plot(
                                dep_feat,
                                shap_values[class_idx],
                                st.session_state.X_test[:100],
                                display_features=st.session_state.X_test_original[:100],
                                feature_names=st.session_state.feature_names,
                                ax=ax,
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    shap.dependence_plot(
                        dep_feat,
                        shap_values_binary,
                        st.session_state.X_test[:100],
                        display_features=st.session_state.X_test_original[:100],
                        feature_names=st.session_state.feature_names,
                        ax=ax,
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
                
                indiv_tabs = st.tabs(["💧 Waterfall Plot", "⚡ Force Plot"])
                
                with indiv_tabs[0]:
                    st.markdown('<div class="explanation-box"><b>How to read (Waterfall):</b> Start from base value (expected model output). Red arrows push prediction higher, blue arrows push lower. Final value is the actual prediction.</div>', unsafe_allow_html=True)
                    
                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        st.info(f"📊 Showing waterfall plots for all {len(shap_values)} classes")
                        class_tabs = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" for i in range(len(shap_values))])
                        
                        for class_idx, class_tab in enumerate(class_tabs):
                            with class_tab:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                base_val = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                                shap.waterfall_plot(
                                    shap.Explanation(
                                        values=shap_values[class_idx][sample_idx],
                                        base_values=base_val,
                                        data=st.session_state.X_test_original.iloc[sample_idx].values,
                                        feature_names=st.session_state.feature_names
                                    ),
                                    show=False
                                )
                                plt.tight_layout()
                                st.pyplot(fig)
                    else:
                        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) == 2 else (explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0])
                        fig, ax = plt.subplots(figsize=(10, 4))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values_binary[sample_idx],
                                base_values=base_val,
                                data=st.session_state.X_test_original.iloc[sample_idx].values,
                                feature_names=st.session_state.feature_names
                            ),
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with indiv_tabs[1]:
                    st.markdown('<div class="explanation-box"><b>How to read (Force):</b> Bold score is the model\'s prediction. Red features push the score higher, blue features push it lower.</div>', unsafe_allow_html=True)
                    
                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        st.info(f"📊 Showing force plots for all {len(shap_values)} classes")
                        class_tabs_force = st.tabs([st.session_state.class_names[i] if st.session_state.class_names is not None else f"Class {i}" for i in range(len(shap_values))])
                        
                        for class_idx, class_tab_f in enumerate(class_tabs_force):
                            with class_tab_f:
                                base_val = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                                fig_f = shap.force_plot(
                                    base_val,
                                    shap_values[class_idx][sample_idx],
                                    st.session_state.X_test.iloc[sample_idx].values,
                                    feature_names=st.session_state.feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                                if fig_f is not None:
                                    st.pyplot(fig_f)
                                else:
                                    st.pyplot(plt.gcf())
                                plt.close('all')
                    else:
                        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) == 2 else (explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0])
                        fig_f = shap.force_plot(
                            base_val,
                            shap_values_binary[sample_idx],
                            st.session_state.X_test.iloc[sample_idx].values,
                            feature_names=st.session_state.feature_names,
                            matplotlib=True,
                            show=False
                        )
                        if fig_f is not None:
                            st.pyplot(fig_f)
                        else:
                            st.pyplot(plt.gcf())
                        plt.close('all')
                
                st.markdown('</div>', unsafe_allow_html=True) # End SHAP card
            except Exception as e:
                st.error(f"❌ SHAP computation error: {str(e)}")
    
    # Tab 3: LIME Analysis
    with tab3:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 🟢 LIME Local Explanations")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Simplifies complex neural logic by "localizing" it for a single instance. 
            Helps verify "What was the model thinking at this exact second?" for specific cases.
        </div>
        """, unsafe_allow_html=True)
        
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
                # Identify categorical features for LIME
                categorical_features_idx = []
                categorical_names = {}
                for i, col in enumerate(st.session_state.feature_names):
                    if col in st.session_state.label_encoders:
                        categorical_features_idx.append(i)
                        categorical_names[i] = st.session_state.label_encoders[col].classes_.tolist()
                
                # Create LIME explainer
                explainer_lime = lime_tabular.LimeTabularExplainer(
                    st.session_state.X_train.values,
                    feature_names=st.session_state.feature_names,
                    class_names=st.session_state.class_names if st.session_state.class_names is not None else [f'Class {i}' for i in range(len(pred_proba_lime))],
                    categorical_features=categorical_features_idx,
                    categorical_names=categorical_names,
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
                
                st.markdown('</div>', unsafe_allow_html=True) # End LIME card
            except Exception as e:
                st.error(f"❌ LIME computation error: {str(e)}")
    
    # Tab 4: Comparison
    with tab4:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Cross-Validation: SHAP vs LIME")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Compares two different auditing methods. 
            Alignment between both methods increases decision confidence. 
            <i>Contradictions may signal model uncertainty on that specific data point.</i>
        </div>
        """, unsafe_allow_html=True)
        
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
                if "LogisticRegression" in str(type(st.session_state.model)):
                    explainer_comp = shap.LinearExplainer(st.session_state.model, st.session_state.X_train)
                else:
                    explainer_comp = shap.TreeExplainer(st.session_state.model)
                
                shap_values_comp_raw = explainer_comp.shap_values(st.session_state.X_test[:100])
                
                if isinstance(shap_values_comp_raw, np.ndarray) and len(shap_values_comp_raw.shape) == 3:
                    shap_values_comp = [shap_values_comp_raw[:, :, i] for i in range(shap_values_comp_raw.shape[2])]
                else:
                    shap_values_comp = shap_values_comp_raw
                
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
        st.markdown('</div>', unsafe_allow_html=True) # End Comparison card
        
    # TAB 5: Counterfactual Analysis
    with tab5:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### 🔀 What-If Counterfactual Reasoning")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Answers "What needs to change to reverse this result?". 
            Provides a roadmap by finding the smallest feature tweaks required to flip the AI's decision.
        </div>
        """, unsafe_allow_html=True)
        cf_idx = st.slider("Select base instance", 0, len(st.session_state.X_test)-1, 0, key="cf_sl")
        base_row = st.session_state.X_test.iloc[cf_idx].copy()
        base_pred = st.session_state.model.predict(base_row.values.reshape(1,-1))[0]
        base_prob = st.session_state.model.predict_proba(base_row.values.reshape(1,-1))[0]
        base_nm = st.session_state.class_names[base_pred] if st.session_state.class_names else str(base_pred)
        
        st.markdown(f'<div class="success-pill">Base prediction: <strong>{base_nm}</strong> ({max(base_prob):.2%} confidence)</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🎛️ Interactive What-If Analysis (Manual Perturbation)")
        st.markdown("Move sliders to see how prediction changes:")
        cf_vals = {}
        slider_cols = st.columns(min(3, len(st.session_state.feature_names)))
        for i, feat in enumerate(st.session_state.feature_names):
            col = slider_cols[i % len(slider_cols)]
            with col:
                if feat in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[feat]
                    options = le.classes_.tolist()
                    base_val_raw = int(base_row[feat])
                    # Ensure base_val is within bounds (should be)
                    base_val_idx = min(max(0, base_val_raw), len(options) - 1)
                    
                    selected_label = st.selectbox(feat, options, index=base_val_idx, key=f"cf_{feat}")
                    cf_vals[feat] = le.transform([selected_label])[0]
                else:
                    feat_min = float(st.session_state.X_train[feat].min())
                    feat_max = float(st.session_state.X_train[feat].max())
                    step = (feat_max-feat_min)/50 if feat_max!=feat_min else 0.01
                    cf_vals[feat] = st.slider(feat, feat_min, feat_max,
                                               float(base_row[feat]),
                                               step=step, key=f"cf_{feat}")
        
        cf_row_manual = pd.DataFrame([cf_vals])[st.session_state.feature_names]
        cf_pred_man = st.session_state.model.predict(cf_row_manual.values)[0]
        cf_prob_man = st.session_state.model.predict_proba(cf_row_manual.values)[0]
        cf_nm_man = st.session_state.class_names[cf_pred_man] if st.session_state.class_names else str(cf_pred_man)
        
        st.markdown("---")
        res1,res2 = st.columns(2)
        with res1:
            st.markdown(f'<div class="metric-box"><h3>Original Prediction</h3><h2 style="text-align: center;">{base_nm}</h2></div>', unsafe_allow_html=True)
        with res2:
            changed = cf_pred_man != base_pred
            st.markdown(f'<div class="metric-box"><h3>{"🎉 Prediction Flipped!" if changed else "Current Prediction"}</h3><h2 style="text-align: center; color: {"#3fb950" if changed else "inherit"};">{cf_nm_man}</h2></div>', unsafe_allow_html=True)
        
        if changed:
            st.markdown('<div class="success-pill">✅ Counterfactual found manually!</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 🤖 Auto Counterfactuals (Powered by DiCE)")
        st.markdown('<div class="info-box">DiCE (Diverse Counterfactual Explanations) provides actionable insights by identifying the <b>minimum required changes</b> to features to flip a model\'s prediction. It uses advanced search algorithms to find diverse pathways—ideal for providing "what-if" guidance to end-users (e.g., "What specific changes would lead to a loan approval?").</div>', unsafe_allow_html=True)
        try:
            import dice_ml
            from dice_ml import Data, Model, Dice
            
            with st.spinner("Finding counterfactuals via DiCE..."):
                data_df = st.session_state.X_train.copy()
                data_df['target'] = st.session_state.y_train
                
                continuous_features = st.session_state.feature_names
                d = Data(dataframe=data_df, continuous_features=continuous_features, outcome_name='target')
                
                # Setup custom sklearn wrapper so DiCE sees a standard model
                class DiceSKLearnModel:
                    def __init__(self, model):
                        self.model = model
                    def predict_proba(self, X):
                        return self.model.predict_proba(X)
                    def predict(self, X):
                        return self.model.predict(X)
                
                m = Model(model=st.session_state.model, backend="sklearn")
                exp = Dice(d, m, method="random")
                
                query_instance = pd.DataFrame([base_row.values], columns=st.session_state.feature_names)
                
                is_binary = len(st.session_state.class_names or []) == 2
                desired_class = "opposite" if is_binary else [c for c in range(len(st.session_state.class_names or [])) if c != base_pred][0]
                
                dice_exp = exp.generate_counterfactuals(
                    query_instance, 
                    total_CFs=3, 
                    desired_class=desired_class
                )
                
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                
                if cf_df is not None and len(cf_df) > 0:
                    st.success("DiCE automatically found the following counterfactuals:")
                    
                    # Inverse transform categorical columns for display
                    display_cf_df = cf_df.copy()
                    for col in display_cf_df.columns:
                        if col in st.session_state.label_encoders:
                            le = st.session_state.label_encoders[col]
                            # Robustly convert float strings to category labels
                            display_cf_df[col] = display_cf_df[col].apply(
                                lambda x: le.inverse_transform([int(round(float(x)))])[0] 
                                if pd.notnull(x) else x
                            )
                        elif col == 'target' and 'target' in st.session_state.label_encoders:
                            le = st.session_state.label_encoders['target']
                            display_cf_df[col] = display_cf_df[col].apply(
                                lambda x: le.inverse_transform([int(round(float(x)))])[0]
                                if pd.notnull(x) else x
                            )
                    
                    st.dataframe(display_cf_df, use_container_width=True)
                else:
                    st.warning("DiCE couldn't find a valid counterfactual for this instance with default bounds.")
                    
        except ImportError:
            st.warning("DiCE-ML library is not installed. Falling back to manual interactive selection above.")
        except Exception as e:
            st.warning(f"DiCE encountered an issue generating auto-counterfactuals: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True) # End Counterfactual card

    # TAB 6: Bias & Fairness Audit
    with tab6:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Algorithmic Fairness Protocol")
        st.markdown('<div class="info-box">Auditing the model for systematic bias. Detects if specific groups are treated unfairly by the decision engine.</div>', unsafe_allow_html=True)
        
        fair_feat = st.selectbox("Select Protected Attribute for Audit", st.session_state.feature_names, key="fair_audit_sel")
        
        if fair_feat:
            # Simple Disparate Impact calculation
            X_tmp = st.session_state.X_test_original.copy()
            X_tmp['pred'] = st.session_state.model.predict(st.session_state.X_test)
            
            groups = [g for g in X_tmp[fair_feat].unique() if pd.notnull(g)]
            if len(groups) > 1:
                st.markdown(f"#### Audit: {fair_feat}")
                metrics_fair = []
                
                for g in groups:
                    group_data = X_tmp[X_tmp[fair_feat] == g]
                    # Logic to find "positive" outcome (class 1 or first unique if numeric)
                    unique_preds = X_tmp['pred'].unique()
                    pos_val = unique_preds[1] if len(unique_preds) > 1 else unique_preds[0]
                    pos_rate = (group_data['pred'] == pos_val).mean()
                    metrics_fair.append({'Group': str(g), 'Selection Rate': pos_rate})
                
                fair_df = pd.DataFrame(metrics_fair)
                
                col_fair1, col_fair2 = st.columns([1, 2])
                with col_fair1:
                    st.dataframe(fair_df, use_container_width=True, hide_index=True)
                    
                    max_rate = fair_df['Selection Rate'].max()
                    min_rate = fair_df['Selection Rate'].min()
                    di_ratio = min_rate / max_rate if max_rate > 0 else 1.0
                    
                    st.metric("Disparate Impact Ratio", f"{di_ratio:.2f}", 
                              help="Ratio of min selection rate to max selection rate. Ideal: > 0.8 (Four-Fifths Rule).")
                    
                    if di_ratio < 0.8:
                        st.error("⚠️ BIAS DETECTED: Disparate Impact ratio is below 0.8 threshold.")
                    else:
                        st.success("✅ COMPLIANCE: No significant disparate impact detected.")
                
                with col_fair2:
                    fig_fair, ax_fair = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=fair_df, x='Group', y='Selection Rate', palette='viridis', ax=ax_fair)
                    ax_fair.set_title('Group Selection Rates')
                    ax_fair.axhline(0.8 * max_rate, color='red', linestyle='--', label='80% Threshold')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_fair)
            else:
                st.info("Insufficient variance in selected feature for fairness audit.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 6: Bias & Fairness Audit
    # Tab 6: Bias Audit
    with tab6:
        st.markdown('<div class="premium-card animate-fade">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Algorithmic Fairness Protocol")
        st.markdown("""
        <div class="info-box">
            <b>Mission Briefing:</b> Moral and regulatory audit of the model. 
            Measures if the AI exhibits systematic unfairness towards protected groups (e.g., Gender, Age, or Region).
        </div>
        """, unsafe_allow_html=True)
        
        fair_feat = st.selectbox("Select Protected Attribute for Audit", st.session_state.feature_names)
        
        if fair_feat:
            # Simple Disparate Impact calculation
            X_tmp = st.session_state.X_test_original.copy()
            X_tmp['pred'] = st.session_state.model.predict(st.session_state.X_test)
            
            groups = X_tmp[fair_feat].unique()
            if len(groups) > 1:
                st.markdown(f"#### Audit: {fair_feat}")
                metrics_fair = []
                
                for g in groups:
                    group_data = X_tmp[X_tmp[fair_feat] == g]
                    pos_rate = (group_data['pred'] == 1).mean() if 1 in X_tmp['pred'].values else (group_data['pred'] == X_tmp['pred'].unique()[0]).mean()
                    metrics_fair.append({'Group': str(g), 'Selection Rate': pos_rate})
                
                fair_df = pd.DataFrame(metrics_fair)
                
                col_fair1, col_fair2 = st.columns([1, 2])
                with col_fair1:
                    st.dataframe(fair_df, use_container_width=True, hide_index=True)
                    
                    # Calculate Disparate Impact Ratio
                    max_rate = fair_df['Selection Rate'].max()
                    min_rate = fair_df['Selection Rate'].min()
                    di_ratio = min_rate / max_rate if max_rate > 0 else 1.0
                    
                    st.metric("Disparate Impact Ratio", f"{di_ratio:.2f}", 
                              help="Ratio of min selection rate to max selection rate. Ideal: > 0.8 (Four-Fifths Rule).")
                    
                    if di_ratio < 0.8:
                        st.error("⚠️ BIAS DETECTED: Disparate Impact ratio is below 0.8 threshold.")
                    else:
                        st.success("✅ COMPLIANCE: No significant disparate impact detected.")
                
                with col_fair2:
                    fig_fair, ax_fair = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=fair_df, x='Group', y='Selection Rate', palette='viridis', ax=ax_fair)
                    ax_fair.set_title('Group Selection Rates')
                    ax_fair.axhline(0.8, color='red', linestyle='--', label='80% Threshold')
                    plt.tight_layout()
                    st.pyplot(fig_fair)
            else:
                st.info("Insufficient variance in selected feature for fairness audit.")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome screen
    domain_class = f"domain-{st.session_state.domain}"
    st.markdown(f"""
    <div class="description-box animate-fade {domain_class}">
        <h1 style='color: white; margin-bottom: 0px;'>🛡️ Operational Protocol: {st.session_state.domain}</h1>
        <p style='font-size: 1.1rem; margin-top: 10px; opacity: 0.9; max-width: 800px; margin-left: auto; margin-right: auto;'>
            A high-fidelity framework for auditing machine learning models. Gain transparency through 
            <strong>SHAP</strong>, <strong>LIME</strong>, and <strong>Counterfactual</strong> signatures. 
            Ensuring precision governance and algorithmic trust.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="data-summary-box animate-fade glow-purple" style="height: 100%;">
            <h2 style='margin-bottom: 0px;'>🎯</h2>
            <h3>Global Auditing</h3>
            <p>Discover longitudinal feature influence and governance metrics across entire datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-summary-box animate-fade glow-blue" style="height: 100%;">
            <h2 style='margin-bottom: 0px;'>🔍</h2>
            <h3>Local Verification</h3>
            <p>Verify specific operational decisions via mathematical LIME and SHAP signatures.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="data-summary-box animate-fade glow-teal" style="height: 100%;">
            <h2 style='margin-bottom: 0px;'>📊</h2>
            <h3>Simulation</h3>
            <p>Perform counterfactual 'What-If' reasoning to probe the model's decision boundaries.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Modern Workflow Guide
    st.markdown("### 🛰️ Operational Workflow")
    
    st.markdown("""
    <div class="workflow-container animate-fade">
        <div class="workflow-step">
            <div class="step-number">01</div>
            <div class="step-title">Intelligence Intake</div>
            <div class="step-desc">Upload your mission-specific CSV data or select a domain sample to initialize the governance protocol.</div>
        </div>
        <div class="workflow-step">
            <div class="step-number">02</div>
            <div class="step-title">Reactor Control</div>
            <div class="step-desc">Define target objectives and tactical features. Configure AI hyperparameters for high-precision training.</div>
        </div>
        <div class="workflow-step">
            <div class="step-number">03</div>
            <div class="step-title">Neural Execution</div>
            <div class="step-desc">Trigger training engine. Monitor real-time performance metrics and strategic operational verdicts.</div>
        </div>
        <div class="workflow-step">
            <div class="step-number">04</div>
            <div class="step-title">XAI Verification</div>
            <div class="step-desc">Audit the decision logic via 6 specialized tabs including SHAP, LIME, and Bias Audit protocols.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    
    st.info("💡 **Tip:** This dashboard works best with classification datasets. Make sure your target variable has discrete classes.")