import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

...

# Apply kNN
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=9) # Default=5. Number of neighbors to use
clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
confusion_matrix_display.plot()
from matplotlib import pyplot as plt
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
class_names = list(map(str, clf.classes_))
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=class_names,
    zero_division=0
))

# Optimize k
from sklearn.model_selection import cross_val_score
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []
cross_validation_roc_auc = []
k_values = range(1, 15, 2)
for i in k_values:
    print('k is:', i)
    clf = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy')
    score = scores.mean()
    cross_validation_accuracies.append(score)
    print('10-fold cross-validaiton accuracy is:', score)

    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print('10-fold cross-validaiton precision is:', precision)

    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print('10-fold cross-validaiton recall is:', recall)

    f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print('10-fold cross-validaiton f1score is:', f1score)

    roc_auc = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='roc_auc').mean()
    cross_validation_roc_auc.append(roc_auc)
    print('10-fold cross-validaiton roc_auc is:', roc_auc)

# Create a graph that shows the overall accuracy for different values of the hyperparameter.
plt.figure(figsize=(10,6))
plt.plot(k_values, cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K_value')
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.show()