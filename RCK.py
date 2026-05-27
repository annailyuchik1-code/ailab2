import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt


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

model_reg = DecisionTreeRegressor(random_state=42)
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

print("=== ДЕРЕВО РЕГРЕССИИ ===")
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("R^2:", r2_score(y_test_reg, y_pred_reg))


X_clf = df.drop(columns=['species'])
y_clf = df['species']

lb = LabelBinarizer()
y_clf_encoded = lb.fit_transform(y_clf)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf
)

model_clf = DecisionTreeClassifier(random_state=42)
model_clf.fit(X_train_clf, y_train_clf)

y_score = model_clf.predict_proba(X_test_clf)

print(y_score)

plt.figure()
for i, class_name in enumerate(lb.classes_):
    fpr, tpr, _ = roc_curve(y_test_clf[:, i], y_score[i][:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая для классификации (Decision Tree)")
plt.legend(loc="lower right")
plt.show()