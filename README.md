# Credit Risk & Loan Default Prediction

## Project Overview
This machine learning project predicts whether a credit card customer is likely to default on their payment in the following month. It uses customer demographic information, credit limits, repayment history, bill amounts, and previous payment amounts.

## Business Objective
Help financial institutions identify potential default risk and support better credit-risk assessment. Model predictions should support human decision-making, not automatically determine loan approval.

## Dataset
- Source: UCI Default of Credit Card Clients dataset
- Records: 30,000 customers
- Original columns: 25, including customer ID and target
- Features used: 23 after removing ID and separating the target
- Target: `default payment next month`
- Missing values: 0

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Excel dataset processing with xlrd

## Project Workflow
1. Load and inspect the dataset.
2. Perform exploratory data analysis (EDA).
3. Visualize default distribution, credit limits, age distribution, and selected feature correlations.
4. Split the data into training and testing sets using stratification.
5. Train Logistic Regression and Random Forest classifiers.
6. Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Save the best model and model comparison results.

## Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 67.97% | 80.05% |
| Default Precision | 36.72% | 55.62% |
| Default Recall | 62.02% | 48.46% |
| Default F1-score | 46.13% | 51.79% |
| ROC-AUC | 0.7081 | 0.7606 |

Random Forest achieved the highest ROC-AUC (0.7606) and accuracy (80.05%) in this experiment. Logistic Regression achieved higher recall for the default class, detecting a larger proportion of actual defaulters at the selected classification threshold.

## Key Findings
- 22.12% of customers in the dataset defaulted the following month.
- Random Forest performed better in ROC-AUC and overall accuracy than Logistic Regression.
- Logistic Regression had higher default-class recall, highlighting the trade-off between identifying more potential defaulters and limiting false alerts.
- Credit-risk decisions should consider the cost of missed defaults, false alerts, fairness, and model calibration.

## Project Structure

```text
Credit_Risk_Loan_Default_Prediction/
├── data/
│   └── raw/
├── src/
│   ├── 01_load_data.py
│   ├── 02_eda.py
│   └── 03_train_model.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── model_comparison.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

Activate the virtual environment, install the required packages, and run:

```bash
python src/01_load_data.py
python src/02_eda.py
python src/03_train_model.py
```

## Limitations
These results are based on one train-test split and the default classification threshold. Further work should include cross-validation, threshold tuning, fairness checks, and evaluation on additional data before any real-world use.

## Conclusion
This project demonstrates a complete introductory credit-risk classification workflow, from data inspection and visualization to model comparison and saving a trained model.
