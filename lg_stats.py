import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

df = pd.read_csv('hospital_readmission_dataset_preprocessed.csv')
df = pd.get_dummies(df, drop_first=True)

num_cols = df.select_dtypes(include=[np.number]).columns.drop('label')
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

df = df.reset_index(drop=True)
X_raw = df.drop(columns=['label'])
y = df['label']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
X_const = sm.add_constant(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_const, y, test_size=0.2, random_state=42)
logit_model = sm.Logit(y_train, X_train).fit(disp=0)

results_df = pd.DataFrame({
    'Coefficient': logit_model.params,
    'Odds Ratio': np.exp(logit_model.params),
    'p-value': logit_model.pvalues
})
conf_int = logit_model.conf_int()
results_df['95% CI Lower'] = np.exp(conf_int[0])
results_df['95% CI Upper'] = np.exp(conf_int[1])

print(f"\nMcFadden's Pseudo R-squared: {logit_model.prsquared:.4f}")
table_to_print = results_df.drop(columns=['Coefficient'])
print(tabulate(table_to_print, headers='keys', tablefmt='grid', floatfmt=".4f"))

plt.figure(figsize=(10, 6))
coef_data = results_df.drop('const')['Coefficient'].sort_values()
coef_data.plot(kind='barh', color='teal')
plt.axvline(0, color='black', linewidth=1)
plt.title('Coefficient Plot (Log-Odds Impact)')
plt.xlabel('Coefficient Value')
plt.tight_layout()

plt.figure(figsize=(10, 8))
plot_df = results_df.drop('const').sort_values('Odds Ratio')
errors = [plot_df['Odds Ratio'] - plot_df['95% CI Lower'],
          plot_df['95% CI Upper'] - plot_df['Odds Ratio']]

plt.errorbar(x=plot_df['Odds Ratio'], y=plot_df.index, xerr=errors, fmt='o',
             color='black', ecolor='firebrick', capsize=3, markersize=6)
plt.axvline(x=1, color='blue', linestyle='--', linewidth=2, label='No Effect (OR = 1)')
plt.title('Forest Plot (Odds Ratios with 95% CI)')
plt.xlabel('Odds Ratio')
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 6))
y_pred_prob = logit_model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()

plt.figure(figsize=(7, 6))
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Readmit', 'Readmit'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.tight_layout()

plt.show()