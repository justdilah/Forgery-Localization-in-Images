import joblib
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns
# Build a Gaussian Classifier

my_data_forged = pd.read_csv("CSV/128/trainingdataset_128_awgn_forged_0_30.csv")
my_data_forged = my_data_forged[:10000]
columns = ['corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4','corr_max_5',
           'corr_max_6', 'corr_max_7', 'corr_max_8', 'eudist_1', 'eudist_2', 'eudist_3', 'eudist_4', 'eudist_5',
           'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy']
X_forged = my_data_forged[columns]
y_forged = my_data_forged['type']

my_data_original = pd.read_csv("CSV/128/trainingdataset_128_awgn_original_0_30.csv")
my_data_original = my_data_original[:10000]
X_original = my_data_original[columns]
y_original = my_data_original['type']


X = pd.concat([X_forged, X_original], axis=0)
y = pd.concat([y_forged, y_original], axis=0)
# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



model = BaggingClassifier(n_estimators=100)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='Forged')
joblib.dump(model,'model/model_128_awgn_30.pkl')
print("Ensemble Accuracy:", accuracy)
print("F1-score:", f1)
