import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Load the data into a df
df = pd.read_csv('diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize The Features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis = 1)) # scaler learns mean and SD of current data
scaled_features = scaler.transform(df.drop('target', axis = 1)) # transforms to have mean of 0 and SD of 1

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1]) # new df of prev ready data, except last column

# Split the Data into train and test
'''
x_train: Features used to train the model.
y_train: Correct answers (targets) used during training.
x_test: Features used for testing the model (unseen by the model during training).
y_test: The actual target values for the test data to measure how well the model predicted.

Stratify is a way to make sure that when you split your data into training and testing sets,
both sets maintain the same proportion of each class (or category) as the original dataset.
'''
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=2) # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Optimize C
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []

# Define specific c_values with the required increments
c_values = [0.01] + [round(0.05 + i * 0.05, 2) for i in range(20)]  # Start with 0.01 and then add increments of 0.05

for i in c_values:
    print('c is:', i)
    clf = LogisticRegression(C=i)  # Update C for each iteration

    # Perform 10-fold cross-validation for accuracy, precision, recall, and f1 score
    accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
    cross_validation_accuracies.append(accuracy)
    print('10-fold cross-validation accuracy is:', accuracy)

    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print('10-fold cross-validation precision is:', precision)

    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print('10-fold cross-validation recall is:', recall)

    f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print('10-fold cross-validation f1 score is:', f1score)

# Create a graph that shows the overall accuracy for different values of the hyperparameter.
plt.figure(figsize=(10,6))
plt.plot(c_values, cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. C-Value')
plt.xlabel('C-value')
plt.ylabel('Accuracy')
plt.show()
