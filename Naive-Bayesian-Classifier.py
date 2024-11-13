import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv('diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

## Standardize the features ##
scaler = StandardScaler()

# calculates mean and SD of features except target. 
# 'fits' scaler to data, meaning, learns parameters needed to standardize features
scaler.fit(df.drop('target', axis=1))

# uses mean and SD from previous step to transform og feature data into standardized values. 
# Mean of 0 and SD of 1 for each feature.
scaled_features = scaler.transform(df.drop('target', axis=1))

# new dataframe with standardized feature values
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])


## Split data into train and test sets ## x represents features (input data) and y represents target variable (output label)
# stratify ensures that the split of the dataset maintains same proportion of classes in both training & testing
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Naive Bayes
clf = GaussianNB(priors=None)
clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
confusion_matrix_display.plot()
plt.show()

# Hyperparameter Optimization
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []
cross_validation_roc_auc = []

# Valid priors (values must sum to 1)
priors = ([0.00, 1.00], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.00, 0.00])

for p in priors:
    print('Priors are:', p)
    
    # Create GaussianNB classifier with current priors
    clf = GaussianNB(priors=p)
    
    # Calculate 5-fold cross-validation accuracy and store the mean
    accuracy = cross_val_score(clf, scaled_features, df['target'], cv=5, scoring='accuracy').mean()
    cross_validation_accuracies.append(accuracy)
    print('5-fold cross-validation accuracy is:', accuracy)
    
    # Calculate and store precision
    precision = cross_val_score(clf, scaled_features, df['target'], cv=5, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print('5-fold cross-validation precision is:', precision)
    
    # Calculate and store recall
    recall = cross_val_score(clf, scaled_features, df['target'], cv=5, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print('5-fold cross-validation recall is:', recall)
    
    # Calculate and store F1-score
    f1score = cross_val_score(clf, scaled_features, df['target'], cv=5, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print('5-fold cross-validation f1score is:', f1score)
    
    # Calculate and store ROC AUC score
    roc_auc = cross_val_score(clf, scaled_features, df['target'], cv=5, scoring='roc_auc').mean()
    cross_validation_roc_auc.append(roc_auc)
    print('5-fold cross-validation ROC AUC is:', roc_auc)

# Plot Accuracy vs Priors
plt.figure(figsize=(10, 6))
plt.plot(['0-100', '10-90', '20-80', '30-70', '40-60', '50-50', '60-40', '70-30', '80-20', '90-10', '100-0'],
         cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. Prior')
plt.xlabel('Prior')
plt.ylabel('Accuracy')
plt.show()
