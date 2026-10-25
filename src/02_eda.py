
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "default_of_credit_card_clients.xls"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_PATH, header=1)

# Remove customer ID because it is not a useful predictive feature
df = df.drop(columns=["ID"])

target = "default payment next month"

print("Dataset shape:", df.shape)
print("\nDefault counts:")
print(df[target].value_counts())
print("\nDefault percentages:")
print((df[target].value_counts(normalize=True) * 100).round(2))
print("\nStatistical summary:")
print(df.describe().round(2))

# 1. Default distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target)
plt.title("Credit Card Default Distribution")
plt.xlabel("Default next month (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "default_distribution.png")
plt.close()

# 2. Credit limit by default status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=target, y="LIMIT_BAL")
plt.title("Credit Limit vs Default Status")
plt.xlabel("Default next month (0 = No, 1 = Yes)")
plt.ylabel("Credit Limit")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "credit_limit_vs_default.png")
plt.close()

# 3. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="AGE", hue=target, bins=25, multiple="layer")
plt.title("Customer Age and Default Status")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.close()

# 4. Correlation heatmap for selected variables
selected = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
    "PAY_AMT1", "PAY_AMT2", target
]
plt.figure(figsize=(10, 7))
sns.heatmap(df[selected].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("\nEDA completed!")
print("Charts saved in:", OUTPUT_DIR)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "default_of_credit_card_clients.xls"
MODEL_DIR = PROJECT_DIR / "outputs" / "models"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_excel(DATA_PATH, header=1)
df = df.drop(columns=["ID"])

target = "default payment next month"
X = df.drop(columns=[target])
y = df[target]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    })

    trained_models[name] = model

    print(classification_report(
        y_test, predictions, target_names=["No Default", "Default"],
        zero_division=0
    ))

    # Save confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["No Default", "Default"],
        cmap="Blues"
    )
    display.ax_.set_title(f"{name} - Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(
        FIGURE_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.close(display.figure_)

# Compare models
results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
print("\nMODEL COMPARISON:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(PROJECT_DIR / "outputs" / "model_comparison.csv", index=False)

# Save the model with the highest ROC AUC
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
joblib.dump(best_model, MODEL_DIR / "best_credit_risk_model.joblib")

print(f"\nBest model by ROC AUC: {best_name}")
print("Model saved to:", MODEL_DIR / "best_credit_risk_model.joblib")
print("Comparison saved to:", PROJECT_DIR / "outputs" / "model_comparison.csv")
print("Training and evaluation completed!")
