# Fraud Detection & Risk Analysis

Machine learning project to detect fraudulent mobile financial transactions, built as a capstone project at the Boston Institute of Analytics.

**Author:** Joshvil Riya John Almeida

## Project Overview
The goal is to build a model that identifies fraudulent transactions accurately, so a company can improve security, reduce financial losses and understand what drives fraud.

## Dataset
- 11,142 transactions, 10 columns, no missing values and no duplicates
- 1,142 fraudulent transactions (10.25%)
- Transaction types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER

## Approach
1. **EDA:** fraud occurs only in TRANSFER (38.5% fraud rate) and CASH_OUT (30.9%); 97.46% of frauds drain the sender's account vs 16.03% of legitimate transactions
2. **Feature engineering:** balance-error features (`errorBalanceOrig`, `errorBalanceDest`), one-hot encoding of transaction type, ID columns dropped
3. **Split and imbalance:** stratified 80/20 split, SMOTE applied to training data only, StandardScaler
4. **Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
5. **Evaluation:** precision, recall, F1-score, ROC-AUC, confusion matrix
6. **Financial impact** analysis and a real-time scoring simulation using the saved model

## Results (XGBoost, 2,229-transaction test set)
| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.996 |
| F1-score | 0.998 |
| ROC-AUC | 1.000 |

Confusion matrix: 227 frauds caught, 2,001 legitimate transactions correctly allowed, 0 false alarms, 1 missed fraud.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 1.000 | 0.996 | 0.998 | 1.000 |
| XGBoost | 1.000 | 0.996 | 0.998 | 1.000 |
| Gradient Boosting | 1.000 | 0.996 | 0.998 | 1.000 |
| Decision Tree | 1.000 | 0.991 | 0.996 | 0.998 |
| Logistic Regression | 1.000 | 0.899 | 0.947 | 0.991 |

## Financial Impact
- The notebook estimates savings using the average fraud amount × number of cases: about $270.7M saved and $1.19M lost to the missed fraud
- The Power BI dashboard uses the actual transaction amounts from the test set: $235.4M caught and $0.42M missed
- False alarm cost: $0

## Business Questions
1. **Precision and accuracy?** Precision 1.000; accuracy 99.96% (2,228 of 2,229 correct)
2. **Reliability?** Recall 0.996, F1 0.998, ROC-AUC 1.000; no legitimate transaction flagged
3. **Losses from errors?** One missed fraud; false alarms cost $0

## Repository Structure
```
notebook/       Jupyter notebook with analysis, preprocessing and model training
data/           Fraud_Analysis_Dataset.csv, powerbi_fraud_data.csv, powerbi_model_comparison.csv
model/          fraud_model.pkl, scaler.pkl
dashboard/      Power BI dashboard (.pbix) and PDF export
presentation/   Project presentation (PowerPoint)
```

## How to Run
1. Install Python with `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `matplotlib` and `seaborn`
2. Place `Fraud_Analysis_Dataset.csv` in the same folder as the notebook (or update the path in the first data-loading cell)
3. Open the notebook in Jupyter and run all cells

## Notes and Limitations
- In this dataset legitimate transactions only have `step` values from 1 to 7, while fraud spans 1 to 95, so `step` is strongly correlated with fraud (0.83). This most likely reflects how the sample was built, so results should be validated on larger real-world data before production use
- The `.pkl` files were created from this notebook; only load pickle files you trust
