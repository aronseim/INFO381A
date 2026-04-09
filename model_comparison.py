import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

df = pd.read_csv('hospital_readmission_dataset_preprocessed.csv')
df = pd.get_dummies(df, drop_first=True)

num_cols = df.select_dtypes(include=[np.number]).columns.drop('label')
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

X = df.drop(columns=['label'])
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )
}
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    cv = cross_validate(model, X_scaled, y, cv=kf, scoring=scoring)

    row = {
        "Model": name,
        "Accuracy": np.mean(cv['test_accuracy']),
        "Precision": np.mean(cv['test_precision']),
        "Recall": np.mean(cv['test_recall']),
        "F1": np.mean(cv['test_f1']),
        "ROC-AUC": np.mean(cv['test_roc_auc'])
    }
    results.append(row)

results_df = pd.DataFrame(results)

print("\nModel Comparison Table (10-Fold Cross-Validation):")
print(tabulate(
    results_df,
    headers='keys',
    tablefmt='grid',
    numalign="center",
    stralign="center",
    floatfmt=".4f"
))