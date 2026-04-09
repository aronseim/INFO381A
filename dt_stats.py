import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tabulate import tabulate

df = pd.read_csv('hospital_readmission_dataset_preprocessed.csv')
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=2,
    random_state=42
)
clf.fit(X_train, y_train)

split_quality = pd.DataFrame({
    'Feature': X.columns,
    'Gini Importance (Split Quality)': clf.feature_importances_
}).sort_values(by='Gini Importance (Split Quality)', ascending=False)

print("\n--- Split Quality Summary (Gini Gain) ---")
print(tabulate(split_quality[split_quality['Gini Importance (Split Quality)'] > 0],
               headers='keys', tablefmt='grid', floatfmt=".4f"))

tree_ = clf.tree_
leaf_indices = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1]
leaf_impurities = [tree_.impurity[i] for i in leaf_indices]

print("\n--- Leaf Purity Analysis ---")
purity_metrics = [
    ["Mean Leaf Gini (Purity)", np.mean(leaf_impurities)],
    ["Best Leaf Gini", np.min(leaf_impurities)],
    ["Worst Leaf Gini", np.max(leaf_impurities)],
    ["Total Leaf Nodes", len(leaf_indices)]
]
print(tabulate(purity_metrics, headers=['Metric', 'Value'], tablefmt='grid', floatfmt=".4f"))

plt.figure(figsize=(20, 10))
plot_tree(clf,
          feature_names=list(X.columns),
          class_names=['No Readmit', 'Readmit'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Diagram")

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
train_acc, test_acc = [], []

for alpha in ccp_alphas:
    temp_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha, max_depth=3)
    temp_clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, temp_clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, temp_clf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_acc, marker='o', label="Train", drawstyle="steps-post")
plt.plot(ccp_alphas, test_acc, marker='o', label="Test", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Pruning Curve")
plt.legend()

plt.figure(figsize=(7, 6))
cm = confusion_matrix(y_test, clf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Readmit', 'Readmit'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix")

plt.figure(figsize=(10, 6))
top_splits = split_quality[split_quality['Gini Importance (Split Quality)'] > 0]
plt.barh(top_splits['Feature'], top_splits['Gini Importance (Split Quality)'], color='teal')
plt.gca().invert_yaxis()
plt.title("Feature Split Summary")
plt.xlabel("Gini Importance")
plt.tight_layout()

plt.show()