# ================================
# Credit Card Default Prediction
# ================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------
# Step 2: Load Dataset
# --------------------------------
# Replace path if needed
df = pd.read_csv("credit_card_default.csv")

# --------------------------------
# Step 3: Basic Data Exploration
# --------------------------------
df.head()
df.info()
df.describe()

# --------------------------------
# Step 4: Data Cleaning
# --------------------------------
# Rename target column if required
if 'default payment next month' in df.columns:
    df.rename(columns={'default payment next month': 'default'}, inplace=True)

# --------------------------------
# Step 5: Feature & Target Split
# --------------------------------
X = df.drop('default', axis=1)
y = df['default']

# --------------------------------
# Step 6: Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------
# Step 7: Feature Scaling
# --------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------
# Step 8: Logistic Regression Model
# --------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# --------------------------------
# Step 9: Decision Tree Model
# --------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# --------------------------------
# Step 10: Random Forest Model
# --------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --------------------------------
# Step 11: Model Evaluation
# --------------------------------
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# --------------------------------
# Step 12: Confusion Matrix & Report
# --------------------------------
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
