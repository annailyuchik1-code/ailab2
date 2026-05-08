import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


df = pd.read_csv("iris.csv")

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)


target_reg = 'petal_length'
df_reg = pd.get_dummies(df, columns=['species'], drop_first=True)
X_reg = df_reg.drop(columns=[target_reg])
y_reg = df_reg[target_reg]

scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("=== ЛИНЕЙНАЯ РЕГРЕССИЯ ===")
print("MSE:", mse)
print("R^2:", r2)


X_clf = df.drop(columns=['species'])
y_clf = df['species']

le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded
)

model_clf = LogisticRegression(max_iter=200)
model_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = model_clf.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
report = classification_report(y_test_clf, y_pred_clf, target_names=le.classes_)

print("\n=== КЛАССИФИКАЦИЯ ===")
print("Accuracy:", accuracy)
print(report)