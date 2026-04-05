import pandas as pd
import numpy as np
from tabulate import tabulate

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df = pd.read_csv("hospital_readmission_dataset.csv")

y = df["label"].to_numpy()
X = df.drop(columns=["label"])

numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns
categorical_features = [c for c in categorical_features if c.lower() not in ["patient_id", "id"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

table = []

for model_name in ["Logistic Regression", "Decision Tree", "SVM"]:
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []
    fold_auc = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == "Logistic Regression":
            model = Pipeline([
                ("preprocess", preprocessor),
                ("model", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ))
            ])
        elif model_name == "Decision Tree":
            model = Pipeline([
                ("preprocess", preprocessor),
                ("model", DecisionTreeClassifier(
                    max_depth=5,
                    class_weight="balanced"
                ))
            ])
        else:
            model = Pipeline([
                ("preprocess", preprocessor),
                ("model", SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced"
                ))
            ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_prec.append(precision_score(y_test, y_pred))
        fold_rec.append(recall_score(y_test, y_pred))
        fold_f1.append(f1_score(y_test, y_pred))
        fold_auc.append(roc_auc_score(y_test, y_prob))

    table.append([
        model_name,
        f"{np.mean(fold_acc):.3f}",
        f"{np.mean(fold_prec):.3f}",
        f"{np.mean(fold_rec):.3f}",
        f"{np.mean(fold_f1):.3f}",
        f"{np.mean(fold_auc):.3f}",
    ])

header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
print(tabulate(table, header, tablefmt="pipe"))
