import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load the data
df = pd.read_csv('diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the Features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df.head())

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Parameters for iteration
min_samples_split_values = [5, 10, 15, 20]
min_samples_leaf_values = [3, 7, 11, 15]

# store results for plotting
decision_tree_accuracies = {split: [] for split in min_samples_split_values}
random_forest_accuracies = {split: [] for split in min_samples_split_values}

# initialize a Decision Tree Classifier for comparison
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=7, min_samples_leaf=2)

# fit the Decision Tree and evaluate metrics
clf.fit(x_train, y_train)
predictions_tree = clf.predict(x_test)

# display confusion matrix for Decision Tree
confusion_matrix_tree = metrics.confusion_matrix(y_test, predictions_tree, labels=clf.classes_)
confusion_matrix_display_tree = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_tree, display_labels=clf.classes_)
confusion_matrix_display_tree.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

# iterate over min_samples_split and min_samples_leaf values for Decision Tree
print("Decision Tree Results:")
for split in min_samples_split_values:
    for leaf in min_samples_leaf_values:
        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=split, min_samples_leaf=leaf, random_state=42)
        
        # Cross-validation for Decision Tree
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        f1 = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        
        # Append accuracy for plotting
        decision_tree_accuracies[split].append(accuracy)
        
        # Print results
        print(f"min_samples_split={split}, min_samples_leaf={leaf} | Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Iterate over min_samples_split and min_samples_leaf values for Random Forest
print("\nRandom Forest Results:")
for split in min_samples_split_values:
    for leaf in min_samples_leaf_values:
        clf = RandomForestClassifier(max_depth=None, min_samples_split=split, min_samples_leaf=leaf, n_estimators=100, random_state=42)
        
        # Fit the Random Forest classifier to retrieve classes_
        clf.fit(x_train, y_train)
        
        # Cross-validation for Random Forest
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        f1 = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        
        # Append accuracy for plotting
        random_forest_accuracies[split].append(accuracy)
        
        # Print results
        print(f"min_samples_split={split}, min_samples_leaf={leaf} | Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# plot Decision Tree accuracy vs min_samples_leaf for each min_samples_split value
plt.figure(figsize=(14, 6))
for split in min_samples_split_values:
    plt.plot(min_samples_leaf_values, decision_tree_accuracies[split], label=f'min_samples_split={split}', marker='o')
plt.title('Decision Tree: Accuracy vs. min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot Random Forest accuracy vs min_samples_leaf for each min_samples_split value
plt.figure(figsize=(14, 6))
for split in min_samples_split_values:
    plt.plot(min_samples_leaf_values, random_forest_accuracies[split], label=f'min_samples_split={split}', marker='o')
plt.title('Random Forest: Accuracy vs. min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


# Produce visualizations of the tree graph
clf = DecisionTreeClassifier(max_depth=None,
                           min_samples_split=7,
                           min_samples_leaf=2)
clf.fit(x_train, y_train)

class_names = list(map(str, clf.classes_))
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree=clf,
    max_depth=3,
    feature_names=feature_names,
    class_names=class_names,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()