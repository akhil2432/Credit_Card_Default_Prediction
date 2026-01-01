# Credit Card Default Prediction

This project focuses on predicting whether a customer will default on their credit card payment using machine learning techniques. It is a binary classification problem widely used in banking and financial risk analysis.

---

## 📌 Objective
To build a machine learning model that predicts credit card default based on customer demographic and financial data.

---

## 📊 Dataset Description
The dataset contains information about customers such as:
- Credit limit
- Gender
- Education
- Marital status
- Age
- Repayment status
- Bill amounts
- Previous payment amounts

**Target Variable**
- `default.payment.next.month`
  - 0 → No Default
  - 1 → Default

---

## 🔄 Project Workflow
1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Result Analysis

---

## 🧠 Machine Learning Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🛠 Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## ▶ How to Run
```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Default.ipynb
✅ Conclusion

The model successfully predicts credit card default risk and helps financial institutions make better credit decisions.
