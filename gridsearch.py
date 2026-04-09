import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('hospital_readmission_dataset_preprocessed.csv')
df = pd.get_dummies(df, drop_first=True)

num_cols = df.select_dtypes(include=[np.number]).columns.drop('label')
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

X = StandardScaler().fit_transform(df.drop(columns=['label']))
y = df['label']

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

dt_param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'criterion': ['gini', 'entropy']
}

print("Searching for best Logistic Regression parameters...")
lr_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_param_grid, cv=5, scoring='f1')
lr_search.fit(X, y)

print("Searching for best Decision Tree parameters...")
dt_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='f1')
dt_search.fit(X, y)

print("\n--- Best Parameters Found ---")
print(f"Logistic Regression: {lr_search.best_params_}")
print(f"Decision Tree: {dt_search.best_params_}")