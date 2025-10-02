import pandas as pd

def preprocess_data(data, target_col):
    # Drop rows with missing target
    data = data.dropna(subset=[target_col])
    
    # Separate target
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    
    # Fill missing values
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])  # Fill with most frequent
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())   # Fill with median
    
    # One-Hot Encode categorical columns only
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y

