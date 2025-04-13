import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('oasis_longitudinal (2).csv')

data_selected = data[["Group", "M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF", "CDR"]]
X = data_selected.drop('Group', axis=1)
y = data_selected['Group']

X_main, X_test, y_main, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.25, random_state=42)

median_ses = X_train['SES'].median()
median_mmse = X_train['MMSE'].median()

X_train['SES'] = X_train['SES'].fillna(median_ses)
X_test['SES'] = X_test['SES'].fillna(median_ses)
X_val['SES'] = X_val['SES'].fillna(median_ses)
X_train['MMSE'] = X_train['MMSE'].fillna(median_mmse)
X_test['MMSE'] = X_test['MMSE'].fillna(median_mmse)
X_val['MMSE'] = X_val['MMSE'].fillna(median_mmse)

label_encoder_MF = LabelEncoder()
X_train['M/F'] = label_encoder_MF.fit_transform(X_train['M/F'])
X_val['M/F'] = label_encoder_MF.transform(X_val['M/F'])
X_test['M/F'] = label_encoder_MF.transform(X_test['M/F'])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

colors = ['#ff7f0e','#2ca02c', '#d62728'] #colors for ROC curves

smote=SMOTE(sampling_strategy='minority', random_state=50)
X_train, y_train = smote.fit_resample(X_train, y_train)

# --------------Decision Tree model--------------------------------
dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, max_features='sqrt', class_weight= {0:1.3, 1:1, 2:1})
dt_model.fit(X_train, y_train)
# Validation
val_dt_pred = dt_model.predict(X_val)
dt_val_acc = accuracy_score(y_val, val_dt_pred)
print("Decision Tree Validation Accuracy:", dt_val_acc)
print(classification_report(y_val, val_dt_pred))
# Testing
y_dt_pred = dt_model.predict(X_test)
dt_test_acc = accuracy_score(y_test, y_dt_pred)
print("Decision Tree Test Accuracy:", dt_test_acc)
print(classification_report(y_test, y_dt_pred))


y_prob = dt_model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
n_classes = y_prob.shape[1]

# micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='#1f77b4', lw=2, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')
# ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

# Diagonal line (random guessing)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.title('ROC Curve for Decision Tree algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


#---------------Random Forest model--------------------------------
rf_model = RandomForestClassifier(random_state=42, class_weight={0:1.1, 1:1, 2:1}, max_depth=9)
rf_model.fit(X_train, y_train)
# Validation
val_rf_pred = rf_model.predict(X_val)
rf_val_acc = accuracy_score(y_val, val_rf_pred)
print("Random Forest Validation Accuracy:", rf_val_acc)
print(classification_report(y_val, val_rf_pred))
# Testing
y_rf_pred = rf_model.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_rf_pred)
print("Random Forest Test Accuracy:", rf_test_acc)
print(classification_report(y_test, y_rf_pred))
# conf_matrix = confusion_matrix(y_test, y_rf_pred)
# print(conf_matrix)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = rf_model.predict_proba(X_test)

plt.figure(figsize=(8, 6))
n_classes = y_prob.shape[1]

# Micro-average line
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='#1f77b4', lw=2, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')

# ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

# Diagonal line (random guessing)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.title('ROC Curve for Random Forest algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#------------------------SVM model----------------------------------------------------
# StandardScaler
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_val_svm = scaler.transform(X_val)
X_test_svm = scaler.transform(X_test)

svm_model = SVC(probability=True, random_state=42, C=2.1)
svm_model.fit(X_train_svm, y_train)
# Validation
val_svm_pred = svm_model.predict(X_val_svm)
svm_val_acc = accuracy_score(y_val, val_svm_pred)
print("SVM Validation Accuracy:", svm_val_acc)
print(classification_report(y_val, val_svm_pred))
# Testing
y_svm_pred = svm_model.predict(X_test_svm)
svm_test_acc = accuracy_score(y_test, y_svm_pred)
print("SVM Test Accuracy:", svm_test_acc)
print(classification_report(y_test, y_svm_pred))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = svm_model.predict_proba(X_test_svm)

plt.figure(figsize=(8, 6))
n_classes = y_prob.shape[1]
# Micro-average line
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='#1f77b4', lw=2, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')
# Diaginal line
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.title('ROC Curve for SVM algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#------------------------------XGBoost model----------------------------------------------------
xgb_model = XGBClassifier(random_state=42, objective='multi:soft', num_class=3, n_estimators=100, learning_rate=0.06, max_depth=8, gamma=0.1, colsample_bytree=0.8)
xgb_model.fit(X_train, y_train)
# Validation
val_xgb_pred = xgb_model.predict(X_val)
xgb_val_acc = accuracy_score(y_val, val_xgb_pred)
print("XGBoost Validation Accuracy:", xgb_val_acc)
print(classification_report(y_val, val_xgb_pred))
# Testing
y_xgb_pred = xgb_model.predict(X_test)
xgb_test_acc = accuracy_score(y_test, y_xgb_pred)
print("XGBoost Test Accuracy:", xgb_test_acc)
print(classification_report(y_test, y_xgb_pred))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = xgb_model.predict_proba(X_test)
plt.figure(figsize=(8, 6))
n_classes = y_prob.shape[1]
# Micro-average line
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='#1f77b4', lw=2, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')
# ROC for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')
# Diagonal line
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.title('ROC Curve XGBoost algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#-------------------------Voting Classifier---------------------------------------
voting = VotingClassifier(
    estimators=[("DT", dt_model), ("RF", rf_model), ("SVM", svm_model), ("XGB", xgb_model)],
    voting="soft"
)

voting.fit(X_train, y_train)
y_vote_pred = voting.predict(X_test)
voting_acc = accuracy_score(y_test, y_vote_pred)
print("Voting Classifier Test Accuracy:", voting_acc)
print(classification_report(y_test, y_vote_pred))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = voting.predict_proba(X_test)
plt.figure(figsize=(8, 6))
n_classes = y_prob.shape[1]
# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='#1f77b4', lw=2, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')
# ROC for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')
# Diagnal line
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.title('ROC Curve for Voting Classifier algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()