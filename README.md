\# 🔍 Explainable AI Dashboard



An interactive Streamlit dashboard for explaining machine learning model predictions using SHAP and LIME techniques.



\## ✨ Features



\- 📊 \*\*Feature Importance Analysis\*\* - Discover which features matter most in your model's decisions

\- 🔵 \*\*SHAP Explanations\*\* - Game theory-based feature attribution with global and local insights

\- 🟢 \*\*LIME Explanations\*\* - Model-agnostic local interpretable explanations

\- ⚖️ \*\*SHAP vs LIME Comparison\*\* - Side-by-side comparison of both explainability methods

\- 📈 \*\*Performance Metrics\*\* - Accuracy, Precision, Recall, F1-Score

\- 🔥 \*\*Correlation Heatmap\*\* - Visualize feature relationships

\- 🎯 \*\*Confusion Matrix\*\* - Detailed classification performance

\- 💡 \*\*Multi-class Support\*\* - Works with binary and multi-class classification problems



\## 🚀 Quick Start



\### Installation



1\. Clone the repository:

```bash

git clone https://github.com/AtamerErkal/explainable\_ai\_dashboard.git

cd explainable\_ai\_dashboard

```



2\. Create a virtual environment (recommended):

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate

```



3\. Install dependencies:

```bash

pip install -r requirements.txt

```



\### Running the Dashboard



```bash

streamlit run dashboard.py

```



The dashboard will open in your browser at `http://localhost:8501`



\## 📖 Usage Guide



\### Step 1: Upload Data

\- Click "Browse files" in the sidebar

\- Upload your CSV file

\- Preview your data



\### Step 2: Configure Model

\- Select target variable (what you want to predict)

\- Choose features to include

\- Select a machine learning algorithm:

&nbsp; - Random Forest

&nbsp; - Gradient Boosting

&nbsp; - Logistic Regression



\### Step 3: Train Model

\- Adjust hyperparameters if needed

\- Set train/test split ratio

\- Click "🚀 Train Model"



\### Step 4: Analyze Results

Explore four powerful analysis tabs:



\#### 🎯 Feature Importance

\- Bar chart showing most important features

\- Top 5 features ranked by importance



\#### 🔵 SHAP Analysis

\- \*\*Summary Plot\*\*: See how each feature impacts predictions across all samples

\- \*\*Bar Plot\*\*: Average absolute impact of features

\- \*\*Waterfall Plot\*\*: Detailed explanation for individual predictions

\- Multi-class support with separate visualizations per class



\#### 🟢 LIME Analysis

\- Local explanations for individual predictions

\- Feature contributions with conditions

\- Multi-class support with tabs for each class



\#### ⚖️ SHAP vs LIME Comparison

\- Side-by-side comparison of both methods

\- Understand strengths and weaknesses

\- Method characteristics summary



\## 📊 Supported Models



\- \*\*Random Forest Classifier\*\* - Ensemble of decision trees

\- \*\*Gradient Boosting Classifier\*\* - Sequential ensemble method

\- \*\*Logistic Regression\*\* - Linear classification model



\## 🔧 Requirements



streamlit==1.27.0

pandas==2.0.3

numpy==1.26.0

scikit-learn==1.3.0

matplotlib==3.8.0

seaborn==0.12.2

shap==0.44.0

lime==0.2.0.1

plotly==5.20.0

numba==0.59.0

llvmlite==0.42.0



See `requirements.txt` for complete list.



\## 📝 Example Datasets



The dashboard works with any classification dataset in CSV format. Example datasets you can try:



\- \*\*Titanic Dataset\*\* - Predict survival on the Titanic

\- \*\*Iris Dataset\*\* - Classify flower species

\- \*\*Wine Quality\*\* - Predict wine quality ratings

\- \*\*Customer Churn\*\* - Predict customer churn



\### Dataset Requirements:

\- CSV format

\- One column as target variable (what to predict)

\- Remaining columns as features

\- Handles both numeric and categorical features

\- Missing values are automatically handled



\## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request.



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📄 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 🙏 Acknowledgments



\- \[SHAP](https://github.com/slundberg/shap) - SHapley Additive exPlanations

\- \[LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations

\- \[Streamlit](https://streamlit.io/) - The fastest way to build data apps

\- \[scikit-learn](https://scikit-learn.org/) - Machine learning in Python



\## 📧 Contact



Atamer Erkal - \[@yourlinkedin](https://www.linkedin.com/in/atamererkal/)



Project Link: \[https://github.com/AtamerErkal/explainable\_ai\_dashboard](https://github.com/AtamerErkal/explainable\_ai\_dashboard)



\## 🎓 Learn More



\- \[Understanding SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)

\- \[Understanding LIME](https://christophm.github.io/interpretable-ml-book/lime.html)

\- \[Explainable AI Guide](https://www.oreilly.com/library/view/interpretable-machine-learning/9781492033158/)



---



⭐ If you find this project helpful, please give it a star!

