# Model Documentation – Credit Card Default Prediction

---

## Model Type
Binary Classification

---

## Input Features
- Credit limit
- Age
- Gender
- Education
- Repayment history
- Bill amount
- Payment amount

---

## Output
- 0 → No Default
- 1 → Default

---

## Algorithms Used
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## Train-Test Split
- Training data: 80%
- Testing data: 20%

---

## Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## Final Model Selection
Random Forest was selected due to better accuracy and generalization.

---

## Limitations
- Imbalanced dataset
- Model depends on historical data quality

---

## Improvements
- SMOTE for imbalance
- Cross-validation
- Advanced models like XGBoost
