import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_graphviz
import graphviz


# CSV dosyasını oku
file_path = 'heart.csv'
df = pd.read_csv(file_path, delimiter=';')  # Sütunların ";" ile ayrıldığını belirt

# '?' değerlerini NaN olarak değiştir
df.replace('?', np.nan, inplace=True)

# Sayısal sütunları seç
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Sayısal sütunlardaki NaN değerleri sütunun ortalaması ile doldur
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))
# Hedef değişkeni belirle
target_column = 'HeartDisease'  # Hedef değişkenin sütun adını kullanın

# Bağımsız değişkenleri (X) ve hedef değişkeni (y) belirle
X = df.drop(target_column, axis=1)
y = df[target_column]

feature_names = []
df_X = pd.DataFrame(X, columns=feature_names)

# feature_names listesini güncelle
feature_names = X.columns.tolist()

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı modelini oluştur
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)

# Modeli test et
y_pred_dt = model_dt.predict(X_test)

# Sınıflandırma performansını değerlendir
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_report_dt = classification_report(y_test, y_pred_dt)

# Sonuçları yazdır
print('Decision Tree Classifier Results:')
print(f'Accuracy: {accuracy_dt}')
print('Classification Report:')
print(classification_report_dt)

from sklearn.tree import export_graphviz
import graphviz

# Ağaç yapısını görselleştir
dot_data = export_graphviz(model_dt, out_file=None, feature_names=feature_names,
                           class_names=y.unique().astype(str),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png")  # Ağacı "decision_tree.png" adında bir dosyaya kaydedebilirsiniz
graph.view("decision_tree")  # Ağacı varsayılan PDF görüntüleyici ile açar


# Random Forest Model
model_rf = RandomForestClassifier(max_depth=5,n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Modeli test et
y_pred_rf = model_rf.predict(X_test)

# Sınıflandırma performansını değerlendir
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# Sonuçları yazdır
print('\nRandom Forest Classifier Results:')
print(f'Accuracy: {accuracy_rf}')
print('Classification Report:')
print(classification_report_rf)

# İlk ağacın ağaç yapısını gösterelim
first_tree = model_rf.estimators_[0]

# Ağaç yapısını görselleştir
dot_data = export_graphviz(first_tree, out_file=None, feature_names=feature_names,
                           class_names=y.unique().astype(str),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("first_tree", format="png")  # Ağacı "first_tree.png" adında bir dosyaya kaydedebilirsiniz
graph.view("first_tree")  # Ağacı varsayılan PDF görüntüleyici ile açar

# Linear Regression Model
# HeartDisease sütununu 0.5 eşik değeriyle ikili sınıflandırmaya dönüştür
y_binary = (y > 0.5).astype(int)

# Eğitim ve test setlerini oluştur
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Linear Regression Model
model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)

# Eğitim seti üzerinde tahmin yap
y_train_pred_log = (model_log.predict(X_train_log) > 0.5).astype(int)

# Test seti üzerinde tahmin yap
y_test_pred_log = (model_log.predict(X_test_log) > 0.5).astype(int)

# Doğruluk (Accuracy) değeri
accuracy_train_log = accuracy_score(y_train_log, y_train_pred_log)
accuracy_test_log = accuracy_score(y_test_log, y_test_pred_log)

# F1 Skoru
f1_train_log = f1_score(y_train_log, y_train_pred_log)
f1_test_log = f1_score(y_test_log, y_test_pred_log)

# Hata Matrisi (Confusion Matrix)
conf_matrix_train_log = confusion_matrix(y_train_log, y_train_pred_log)
conf_matrix_test_log = confusion_matrix(y_test_log, y_test_pred_log)

# Sınıflandırma Raporu (Classification Report)
class_report_train_log = classification_report(y_train_log, y_train_pred_log)
class_report_test_log = classification_report(y_test_log, y_test_pred_log)

# Sonuçları yazdır
print('\nLinear Regression Results:')
print(f'Training Accuracy: {accuracy_train_log}')
print(f'Test Accuracy: {accuracy_test_log}')
print(f'Training F1 Score: {f1_train_log}')
print(f'Test F1 Score: {f1_test_log}')

print('Training Classification Report:')
print(class_report_train_log)
print('Test Classification Report:')
print(class_report_test_log)

y_pred = model_log.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')

# SVM Model ve ROC Eğrisi
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Sonuçları yazdır
print(f'Accuracy: {accuracy}')
print('SVM Classification Report:')
print(classification_report_result)
# Decision Tree için ROC Eğrisi
y_score_dt = model_dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_score_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Random Forest için ROC Eğrisi
y_score_rf = model_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

y_probs = svm_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)



# AUC (Alan Altındaki Eğri) değerini hesapla
auc_score = roc_auc_score(y_test, y_probs)
print(f'AUC Score: {auc_score}')


# ROC Eğrisini Çizme
plt.figure(figsize=(8, 8))
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label=f'DecisionTree ROC Curve (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'RandomForest ROC Curve (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr, tpr, color='blue', lw=2, label=f'SVM ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



