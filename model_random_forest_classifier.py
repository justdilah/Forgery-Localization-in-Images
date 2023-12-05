import pandas as pd

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns

my_data_forged = pd.read_csv("late_fusion_fake_1.csv")
my_data_forged = my_data_forged[:10000]
columns = ['left_eye_classifier', 'right_eye_classifier', 'nose_classfier', 'mouth_classifier']

# columns = ['corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4','corr_max_5',
#             'corr_max_6','corr_max_7', 'eudist_1', 'eudist_2', 'eudist_3', 'eudist_4', 'eudist_5',
#            'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy']

X_forged = my_data_forged[columns]
y_forged = my_data_forged['label']

my_data_original = pd.read_csv("late_fusion_real_1.csv")
my_data_original = my_data_original[:10000]
X_original = my_data_original[columns]
y_original = my_data_original['label']


X = pd.concat([X_forged, X_original], axis=0)
y = pd.concat([y_forged, y_original], axis=0)
# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# standardize the range of values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Random Forest classifier
forest_model = RandomForestClassifier(n_estimators=100, random_state=1)
forest_model.fit(X_train, y_train)
forest_y_pred = forest_model.predict(X_test)
# evaluate performance
forest_accuracy = accuracy_score(y_test, forest_y_pred)
forest_precision = precision_score(y_test, forest_y_pred, pos_label='real')
forest_recall = recall_score(y_test, forest_y_pred, pos_label='real')
forest_f1 = f1_score(y_test, forest_y_pred, pos_label='real')
print("Random Forest Accuracy:", forest_accuracy)
print("Random Forest Precision:", forest_precision)
print("Random Forest Recall:", forest_recall)
print("Random Forest F1 Score:", forest_f1)