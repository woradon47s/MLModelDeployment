import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset (assuming you have a dataset called 'Housing.csv')
data = pd.read_csv('Housing.csv')

# Ensure the dataset contains the target column 'price'
if 'price' not in data.columns:
    raise ValueError("The dataset must have a 'price' column as the target variable.")

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns

# Create transformers for imputing missing values in categorical and numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_columns),
        ('cat', Pipeline([
            ('imputer', categorical_imputer),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncode categorical features
        ]), categorical_columns)
    ])

# Create the full pipeline combining the preprocessor and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X, y)

# Save the entire pipeline (model + preprocessor) to a single file
with open('reg_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Model and preprocessor saved as reg_model.pkl")
