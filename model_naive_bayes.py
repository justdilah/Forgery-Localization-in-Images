from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns
# Build a Gaussian Classifier

my_data_forged = pd.read_csv("CSV/128/trainingdataset_128_jp2_forged_0.csv")
my_data_forged = my_data_forged[:1000]
columns = ['corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4','corr_max_5',
           'corr_max_6', 'corr_max_7', 'corr_max_8', 'eudist_1', 'eudist_2', 'eudist_3', 'eudist_4', 'eudist_5',
           'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy']
X_forged = my_data_forged[columns]
y_forged = my_data_forged['type']

my_data_original = pd.read_csv("CSV/128/trainingdataset_128_jp2_original_0.csv")
my_data_original = my_data_original[:1000]
X_original = my_data_original[columns]
y_original = my_data_original['type']



X = pd.concat([X_forged, X_original], axis=0)
y = pd.concat([y_forged, y_original], axis=0)
# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# standardize the range of values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()

# Model training
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("NB Accuracy:", accuracy)
