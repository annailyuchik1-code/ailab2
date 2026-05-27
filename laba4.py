import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report

df = pd.read_csv("iris.csv")

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop(columns=['species'])
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("OOB Score:", rf.oob_score_)
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))


ada = AdaBoostClassifier(random_state=42)

ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

print("\n=== ADABOOST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada, target_names=le.classes_))


gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("\n=== GRADIENT BOOSTING ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb, target_names=le.classes_))

plt.figure()

# Для ROC используем вероятности
models = {
    "Random Forest": rf,
    "AdaBoost": ada,
    "Gradient Boosting": gb
}

for name, model in models.items():
    y_score = model.predict_proba(X_test)

    # Берем класс 1 (для бинарной ROC — упрощение)
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Диагональ (случайная модель)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые")
plt.legend()

plt.show()
