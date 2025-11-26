import pandas as pd
import numpy as np
import pickle
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
#import joblib
df = pd.read_csv(r'C:\Users\LEGION\Desktop\programing\cybersecurity_intrusion_data.csv')
df.head()
print(df)
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df['attack_detected'].value_counts(normalize=True)*100
df = df.drop(columns=["session_id"])
df = df.drop(columns=["encryption_used"])
df = df.drop(columns=["browser_type"])
df = df.drop(columns=["protocol_type"])
from sklearn.model_selection import train_test_split

X = df.drop('attack_detected', axis=1)
y = df['attack_detected']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.head())
print(X_test.size)

print(y_train.head())
print(y_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500)
ann.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "KNN": knn
}

for name, model in models.items():
    pred = model.predict(X_test)
    print("======= ", name, " =======")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

for name, model in models.items():
    filename = name.replace(" ", "_").lower() + ".pkl"
    pickle.dump(model, open(filename, "wb"))
    print(f"Saved: {filename}")

# 1. Predict
y_pred = model.predict(X_test)

# 2. Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

# 3. Classification Report
print(classification_report(y_test, y_pred))

# 4. Plotting
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ['Normal', 'Attack']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='red')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


