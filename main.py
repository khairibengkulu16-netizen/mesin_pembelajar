import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
n_samples = 200

ipk = np.random.uniform(2.0, 3.9, n_samples)
ips_last = ipk + np.random.normal(0, 0.25, n_samples)
ips_last = np.clip(ips_last, 1.5, 4.0)
sks_lulus = np.random.randint(80, 144, n_samples)
presensi = np.random.uniform(50, 100, n_samples)
status_kerja = np.random.choice(['Bekerja', 'Tidak Bekerja'], n_samples, p=[0.35, 0.65])

score = (ipk * 1.8) + (sks_lulus / 30.0) + (presensi / 25.0) - (status_kerja == 'Bekerja') * 1.5
status_lulus = np.where(score > 10.5, 'Tepat Waktu', 'Terlambat')

df = pd.DataFrame({
    'IPK': ipk,
    'IPS_Terakhir': ips_last,
    'SKS_Lulus': sks_lulus,
    'Presensi_Pct': presensi,
    'Status_Kerja': status_kerja,
    'Status_Lulus': status_lulus
})

df.loc[np.random.choice(df.index, 10, replace=False), 'Presensi_Pct'] = np.nan
df.loc[np.random.choice(df.index, 5, replace=False), 'IPK'] = np.nan

df['IPK'] = df['IPK'].fillna(df['IPK'].median())
df['Presensi_Pct'] = df['Presensi_Pct'].fillna(df['Presensi_Pct'].mean())

le_kerja = LabelEncoder()
df['Status_Kerja_Enc'] = le_kerja.fit_transform(df['Status_Kerja'])

le_target = LabelEncoder()
df['Target'] = le_target.fit_transform(df['Status_Lulus'])

feature_cols = ['IPK', 'IPS_Terakhir', 'SKS_Lulus', 'Presensi_Pct', 'Status_Kerja_Enc']
X = df[feature_cols]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth3.fit(X_train, y_train)

dt_none = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_none.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

models = {
    'DT (max_depth=3)': dt_depth3,
    'DT (max_depth=None)': dt_none,
    'Gaussian Naive Bayes': gnb
}

metrics_list = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    metrics_list.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'CM': cm
    })

eval_df = pd.DataFrame(metrics_list)
print(eval_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])

for item in metrics_list:
    print(f"\n{item['Model']} Confusion Matrix:")
    print(item['CM'])

plt.figure(figsize=(16, 8))
plot_tree(
    dt_depth3, 
    feature_names=feature_cols, 
    class_names=le_target.classes_, 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.show()

df_plot = eval_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                       var_name='Metrik', value_name='Skor')

plt.figure(figsize=(10, 5))
sns.barplot(data=df_plot, x='Metrik', y='Skor', hue='Model')
plt.ylim(0.5, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

y_pred_dt3 = dt_depth3.predict(X_test)
err_mask = (y_test != y_pred_dt3)

error_df = df.loc[X_test[err_mask].index, ['IPK', 'IPS_Terakhir', 'SKS_Lulus', 'Presensi_Pct', 'Status_Kerja', 'Status_Lulus']].copy()
error_df['Prediksi_Model'] = le_target.inverse_transform(y_pred_dt3[err_mask])

print("\n")
print(error_df.head(5).to_string())