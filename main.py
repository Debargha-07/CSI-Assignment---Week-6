import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Make output directory
os.makedirs('outputs', exist_ok=True)

# Step 1: Load dataset
print(" Loading Breast Cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

results = []

print(" Training and evaluating models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

    # Classification Report
    with open("outputs/classification_report.txt", "a") as f:
        f.write(f"\n\n===== {name} =====\n")
        f.write(classification_report(y_test, y_pred))

# Step 5: Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/model_performance_summary.csv", index=False)
print("\n Model evaluation completed. Summary saved.\n")
print(df_results)

# Step 6: GridSearchCV for Random Forest
print("\n Performing GridSearchCV on Random Forest...")
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best Parameters (Random Forest):", grid_search_rf.best_params_)

# Step 7: RandomizedSearchCV for SVM
print("\n Performing RandomizedSearchCV on SVM...")
param_dist_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
random_search_svm = RandomizedSearchCV(SVC(probability=True), param_distributions=param_dist_svm,
                                        n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
random_search_svm.fit(X_train, y_train)
best_svm = random_search_svm.best_estimator_
print("Best Parameters (SVM):", random_search_svm.best_params_)

# Step 8: Feature Importance (Random Forest)
print(" Plotting feature importance...")
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Top Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("outputs/feature_importance_rf.png")
plt.close()

# Step 9: ROC Curve (Random Forest)
print(" Generating ROC Curve...")
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve_rf.png")
plt.close()

# Step 10: Model Comparison Plot
print(" Saving model comparison bar plot...")
df_plot = df_results.set_index('Model')
df_plot.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

print(" All tasks completed. Check the 'outputs' folder for results.")
