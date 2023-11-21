import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache()
def load_data():
    """This function returns the preprocessed data"""
    df = pd.read_csv('C:/Users/chand/OneDrive/Desktop/heart_failure_dataset.csv')
    df.columns = ["age", "anaemia","creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine","serum_sodium","sex","smoking","time","DEATH_EVENT"]
    X = df[["age", "anaemia","creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine","serum_sodium","sex","smoking","time","DEATH_EVENT"]]
    y = df['DEATH_EVENT']
    print("DataFrame Shape:", df.shape)
    print("Column Names:", df.columns)

    return df, X, y

@st.cache()
def train_model(X, y):
    """This function trains the model and returns the model and model score"""
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(X, y)
    score = model.score(X, y)
    return model, score

def predict(model, features):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction

# Usage example
df, X, y = load_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
dt_model, dt_score = train_model(X_train, y_train)

# Make predictions on the test set
test_features = X_test.iloc[0, :].values  # Example: using the first row of the test set
dt_prediction = predict(dt_model, test_features)

# Display the results
print("Decision Tree Model Score:", dt_score)
print("Decision Tree Predicted Class:", dt_prediction)
