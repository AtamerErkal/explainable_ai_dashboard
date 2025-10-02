\# 🔍 Explainable AI Dashboard



!\[Python](https://img.shields.io/badge/Python-3.11-blue)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.27.0-orange)

!\[License](https://img.shields.io/badge/License-MIT-green)

!\[Status](https://img.shields.io/badge/Status-Active-brightgreen)



An interactive Streamlit dashboard for explaining machine learning model predictions using SHAP and LIME techniques.



---



\## 📷 Dashboard Preview



!\[Dashboard Screenshot](images/dashboard\_preview.png)



---



\## ✨ Features



\- 📊 \*\*Feature Importance Analysis\*\* — Discover which features matter most in your model's decisions  

\- 🔵 \*\*SHAP Explanations\*\* — Game theory-based feature attribution with global and local insights  

\- 🟢 \*\*LIME Explanations\*\* — Model-agnostic local interpretable explanations  

\- ⚖️ \*\*SHAP vs LIME Comparison\*\* — Side-by-side comparison of both explainability methods  

\- 📈 \*\*Performance Metrics\*\* — Accuracy, Precision, Recall, F1-Score  

\- 🔥 \*\*Correlation Heatmap\*\* — Visualize feature relationships  

\- 🎯 \*\*Confusion Matrix\*\* — Detailed classification performance  

\- 💡 \*\*Multi-class Support\*\* — Works with binary and multi-class classification problems  



---



\## 🚀 Quick Start



```bash

git clone https://github.com/AtamerErkal/explainable\_ai\_dashboard.git

cd explainable\_ai\_dashboard

python -m venv venv

venv\\Scripts\\activate  # On Windows

pip install -r requirements.txt

streamlit run dashboard.py





Open your browser at: http://localhost:8501



📖 Usage Guide

Step 1: Upload Data

\- Click "Browse files" in the sidebar

\- Upload your CSV file

\- Preview your data

Step 2: Configure Model

\- Select target variable

\- Choose features to include

\- Select a machine learning algorithm:

\- Random Forest

\- Gradient Boosting

\- Logistic Regression

Step 3: Train Model

\- Adjust hyperparameters

\- Set train/test split ratio

\- Click "🚀 Train Model"

Step 4: Analyze Results

Explore four powerful analysis tabs:

🎯 Feature Importance

\- Bar chart showing most important features

\- Top 5 features ranked by importance

🔵 SHAP Analysis

\- Summary Plot: Feature impact across all samples

\- Bar Plot: Average absolute impact

\- Waterfall Plot: Individual prediction explanations

\- Multi-class support with per-class visualizations

🟢 LIME Analysis

\- Local explanations for individual predictions

\- Feature contributions with conditions

\- Multi-class support with tabs for each class

⚖️ SHAP vs LIME Comparison

\- Side-by-side comparison

\- Understand strengths and weaknesses

\- Method characteristics summary



📊 Supported Models

\- Random Forest Classifier — Ensemble of decision trees

\- Gradient Boosting Classifier — Sequential ensemble method

\- Logistic Regression — Linear classification model



🔧 Requirements

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





See requirements.txt for full list.



📝 Example Datasets

Works with any classification dataset in CSV format. Try:

\- Titanic Dataset — Predict survival

\- Iris Dataset — Classify flower species

\- Wine Quality — Predict wine ratings

\- Customer Churn — Predict customer attrition

Dataset Requirements:

\- CSV format

\- One target column

\- Remaining columns as features

\- Supports numeric and categorical features

\- Handles missing values automatically



🤝 Contributing

Contributions are welcome!

\# Fork and clone

git checkout -b feature/AmazingFeature

git commit -m "Add AmazingFeature"

git push origin feature/AmazingFeature





Then open a Pull Request.



📄 License

This project is licensed under the MIT License. See the LICENSE file for details.



🙏 Acknowledgments

\- SHAP

\- LIME

\- Streamlit

\- scikit-learn



📧 Contact

Atamer Erkal

LinkedIn Profile

Project Link: GitHub Repository



🎓 Learn More

\- SHAP Explained

\- LIME Explained

\- Interpretable ML Guide



⭐ If you find this project helpful, please give it a star!

