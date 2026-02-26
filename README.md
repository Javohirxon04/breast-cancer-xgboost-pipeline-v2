## XGBoost Pipeline – Step by Step Explanation

Bu hujjat **XGBoost + Pipeline Breast Cancer Classification** loyihasining **qatorma-qator tushuntirishi** hisoblanadi.

---

# 1️⃣ Kutubxonalar (Imports)

```python
import numpy as np
```

**NumPy** — sonli hisob-kitoblar uchun ishlatiladi.

Bu loyihada asosan:

```
np.argsort()
```

feature importance qiymatlarini saralash uchun ishlatiladi.

---

```python
from sklearn.datasets import load_breast_cancer
```

`load_breast_cancer()` — sklearn ichidagi tayyor **Breast Cancer datasetni yuklaydi.**

---

```python
from sklearn.model_selection import train_test_split
```

Datasetni:

* Train set
* Test set

ga bo‘lib beradi.

---

```python
from sklearn.pipeline import Pipeline
```

Pipeline bir nechta bosqichni bitta tizimga bog‘laydi:

```
Scaler → Model
```

Pipeline **data leakage ni oldini oladi.**

---

```python
from sklearn.preprocessing import StandardScaler
```

Featurelarni standartlashtiradi.

Formula:

```
z = (x - mean) / std
```

Natija:

* O‘rtacha ≈ 0
* Standart og‘ish ≈ 1

---

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
```

Modelni baholash uchun ishlatiladi.

### Accuracy

Umumiy to‘g‘ri bashorat foizi.

### Precision

"1 deb aytilganlarning nechasi to‘g‘ri"

### Recall

"Aslida 1 bo‘lganlarning nechasi topildi"

---

```python
from xgboost import XGBClassifier
```

XGBoost klassifikatsiya modeli.

---

# 2️⃣ Datasetni Yuklash

```python
data = load_breast_cancer()
```

Datasetni yuklaydi.

Dataset ichida:

```
data.data
```

Featurelar.

```
data.target
```

Target qiymatlar.

```
data.feature_names
```

Feature nomlari.

---

```python
X = data.data
y = data.target
feature_names = data.feature_names
```

### X

Model kirish qiymatlari.

30 ta feature mavjud.

---

### y

Target label.

```
0 = Malignant
1 = Benign
```

---

### feature_names

Feature nomlari.

Keyinchalik **Top Features chiqarish uchun kerak.**

---

# 3️⃣ Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Dataset bo‘linadi:

```
80% Train
20% Test
```

---

## test_size=0.2

20% test data.

---

## random_state=42

Har safar run qilinganda bir xil natija chiqadi.

Bu **reproducibility** deyiladi.

---

## stratify=y

Classlar nisbatini saqlaydi.

Masalan:

Agar datasetda:

```
60% class 1
40% class 0
```

Train va testda ham shunga yaqin bo‘ladi.

Classification uchun muhim.

---

## Natija

```
X_train , y_train → Training
X_test , y_test → Testing
```

---

# 4️⃣ Pipeline Yaratish

```python
pipe = Pipeline(steps=[

    ("scaler", StandardScaler()),

    ("xgb", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))

])
```

Pipeline ketma-ketligi:

```
StandardScaler
↓
XGBClassifier
```

---

## Pipeline qanday ishlaydi?

### pipe.fit()

Pipeline quyidagilarni qiladi:

1.

```
scaler.fit(X_train)
```

Train datadan mean va std topadi.

---

2.

```
scaler.transform(X_train)
```

Train datani standartlashtiradi.

---

3.

```
xgb.fit(...)
```

Modelni o‘qitadi.

---

### pipe.predict()

Pipeline quyidagilarni qiladi:

1.

```
scaler.transform(X_test)
```

Test datani standartlashtiradi.

---

2.

```
xgb.predict()
```

Predict qiladi.

---

Pipeline **data leakage ni oldini oladi.**

---

# 5️⃣ XGBClassifier Parametrlari

### n_estimators = 300

300 ta daraxt quriladi.

Boosting bosqichlari soni.

---

### learning_rate = 0.05

Har daraxtning hissasi kichik bo‘ladi.

Sekinroq o‘rganadi lekin barqarorroq.

---

### max_depth = 4

Daraxt chuqurligi.

Juda katta bo‘lsa overfitting bo‘ladi.

---

### subsample = 0.9

Har daraxt uchun dataning 90% ishlatiladi.

Overfitting kamayadi.

---

### colsample_bytree = 0.9

Har daraxt uchun featurelarning 90% ishlatiladi.

---

### random_state = 42

Natijani bir xil qiladi.

---

### eval_metric = "logloss"

Baholash metrikasi.

Warning chiqmasligi uchun ham kerak.

---

### n_jobs = -1

Barcha CPU ishlatiladi.

Tezroq ishlaydi.

---

# 6️⃣ Model Training

```python
pipe.fit(X_train, y_train)
```

Pipeline quyidagilarni bajaradi:

```
scaler.fit(X_train)
```

↓

```
scaler.transform(X_train)
```

↓

```
xgb.fit(...)
```

---

# 7️⃣ Prediction

```python
y_pred = pipe.predict(X_test)
```

Pipeline quyidagilarni qiladi:

```
scaler.transform(X_test)
```

↓

```
xgb.predict()
```

Natija:

```
y_pred
```

0 yoki 1 bashoratlar.

---

# 8️⃣ Metrics

```python
acc = accuracy_score(y_test, y_pred)
```

Accuracy:

Umumiy to‘g‘ri bashorat foizi.

---

```python
prec = precision_score(y_test, y_pred)
```

Precision:

1 deb aytilganlarning nechasi to‘g‘ri.

---

```python
rec = recall_score(y_test, y_pred)
```

Recall:

Aslida 1 bo‘lganlarning nechasi topildi.

---

```python
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
```

Natijani 4 xonagacha chiqaradi.

---

# 9️⃣ Feature Importance

```python
model = pipe.named_steps["xgb"]
```

Pipeline ichidan model olinadi.

Pipeline bosqichlari lug‘ati:

```
named_steps
```

---

```python
importances = model.feature_importances_
```

Har bir feature importance qiymatini beradi.

Uzunligi:

```
30 ta feature
```

---

# 🔟 Top 5 Feature

```python
top5_idx = np.argsort(importances)[-5:][::-1]
```

### np.argsort(importances)

Importance qiymatlarini kichikdan kattaga saralaydi.

Indexlarni qaytaradi.

---

### [-5:]

Eng katta 5 ta qiymat olinadi.

---

### [::-1]

Teskari aylantiriladi.

Kattadan kichikka.

---

```python
top5_features = feature_names[top5_idx]
```

Top feature nomlari olinadi.

---

```python
print("\nTop 5 muhim featurelar:")

for i, idx in enumerate(top5_idx, 1):

    print(f"{i}) {feature_names[idx]} (importance={importances[idx]:.4f})")
```

### enumerate(...,1)

Sanash 1 dan boshlanadi.
Natija

1) worst perimeter
2) worst concave points
3) mean concave points
4) worst radius
5) mean perimeter
# 🎯 Xulosa

Bu loyiha quyidagilarni ko‘rsatadi:
* Pipeline ishlatish
* StandardScaler
* XGBoost
* Train/Test Split
* Accuracy
* Precision
* Recall
* Feature Importance


## XGBoost Pipeline – Breast Cancer Classification

---

## 📌 Project Overview

This project demonstrates a **machine learning classification pipeline** using **XGBoost** on the **Breast Cancer dataset** from Scikit-learn.

The goal is to:

* Build a **Pipeline**
* Perform **Train/Test Split**
* Train an **XGBoost model**
* Evaluate the model
* Extract **Top Important Features**

---

## 📊 Dataset

Dataset used:

```
sklearn.datasets.load_breast_cancer()
```

Dataset contains:

* 569 samples
* 30 numerical features
* Binary classification target:

  * 0 = Malignant
  * 1 = Benign

---

## ⚙️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* NumPy

---

## 📦 Import Libraries

```python
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier
```

---

## 📥 Load Dataset

```python
data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
```

---

## ✂️ Train Test Split

Dataset is split into:

* 80% Training Data
* 20% Test Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 🔄 Pipeline Creation

Pipeline consists of:

1. StandardScaler
2. XGBClassifier

```python
pipe = Pipeline(steps=[

    ("scaler", StandardScaler()),

    ("xgb", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))

])
```

---

## 🧠 Model Training

```python
pipe.fit(X_train, y_train)
```

Pipeline automatically:

* Fits scaler on training data
* Transforms training data
* Trains XGBoost model

---

## 🔮 Prediction

```python
y_pred = pipe.predict(X_test)
```

---

## 📈 Model Evaluation

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

### Precision

```python
precision_score(y_test, y_pred)
```

### Recall

```python
recall_score(y_test, y_pred)
```

Example output:

```
Accuracy : 0.9737
Precision: 0.9726
Recall   : 0.9861
```

---

## ⭐ Feature Importance

Extract XGBoost model from Pipeline:

```python
model = pipe.named_steps["xgb"]
```

Get importance values:

```python
importances = model.feature_importances_
```

Top 5 features:

```python
top5_idx = np.argsort(importances)[-5:][::-1]

for i, idx in enumerate(top5_idx, 1):
    print(feature_names[idx], importances[idx])
```

---

## 🥇 Top Important Features (Example)

```
worst perimeter
worst concave points
mean concave points
worst radius
mean perimeter
```

---

## 🚀 How to Run

Install libraries:

```
pip install xgboost scikit-learn numpy
```

Run:

```
python main.py
```

---

## 📌 Key Concepts Demonstrated

* Pipeline
* StandardScaler
* XGBoost
* Train/Test Split
* Accuracy
* Precision
* Recall
* Feature Importance

---

## 👨‍💻 Author

Machine Learning Student
