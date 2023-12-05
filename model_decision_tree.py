
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,auc

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
for feat, importance in zip(X.columns, clf.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))

knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred, pos_label='fake')
knn_recall = recall_score(y_test, y_pred, pos_label='fake')
knn_f1 = f1_score(y_test, y_pred, pos_label='fake')
print("Decision Tree Accuracy:", knn_accuracy)
print("Random Forest Precision:", knn_precision)
print("Random Forest Recall:", knn_recall)
print("Random Forest F1 Score:", knn_f1)
# print("KNN Precision:", knn_precision)
# print("KNN Recall:", knn_recall)
# print("KNN F1 Score:", knn_f1)
# importance = clf.feature_importances_
# pyplot.bar([x for x in X.columns], importance)
# pyplot.show()
