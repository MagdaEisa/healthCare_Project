import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("cleaned_data.csv", low_memory=False)

df.columns = df.columns.str.strip()

print("Missing values:\n", df.isnull().sum())

df = df.drop(columns=["ID", "filepath", "filename", "Left-Fundus", "Right-Fundus",
                      "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"])

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

corr = df[numeric_cols].corr()

print(corr)

numeric_cols = ['Patient Age', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if df['target'].dtype in ['int64', 'float64']:
    numeric_cols.append('target')


df['target'] = df['target'].apply(lambda x: eval(x)[0])  

corr = df[numeric_cols].corr()

for col in df.columns:
    print(f"Column: {col}, Unique Values: {df[col].unique()}")

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
corr = df[numeric_cols + ['target']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix with Target")
plt.show()

X_chi2 = df[numeric_cols]
y_chi2 = df["target"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_chi2)

chi2_vals, p_vals = chi2(X_scaled, y_chi2)
chi2_df = pd.DataFrame({
    'Feature': X_chi2.columns,
    'Chi2 Score': chi2_vals,
    'P-value': p_vals
}).sort_values(by="Chi2 Score", ascending=False)

print("\nChi-square Test Results:")
print(chi2_df)

X = df[numeric_cols + ['Patient Sex_Female', 'Patient Sex_Male']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nRandom Forest Feature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()
