import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Load the data
df = pd.read_csv('diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the Features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df.head())

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)


## Scatter plot of training samples and SVM classifier in a 2d space. ##
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Reduce the number of features to 2, so you can plot them
pca = PCA(n_components=2) # Create a PCA Object that will generate two features from the existing features
pca = pca.fit(x_train) # Fit PCA to training data
x_train_2 = pca.transform(x_train)
x_test_2 = pca.transform(x_test)


## Apply SVM ##
from sklearn.svm import SVC
# clf_pca = SVC(C=1.0, kernel='linear')
# clf_pca = SVC(C=1.0, kernel='poly', degree=3)
# clf_pca = SVC(C=1.0, kernel='sigmoid')
# clf_pca = SVC(C=1.0, kernel='rbf')
# clf_pca = clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)


## Apply Decision Tree Classifier ##
from sklearn import tree
# clf_pca = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=7, min_samples_leaf=2)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

## Apply Random Forest ##
from sklearn.ensemble import RandomForestClassifier
# clf_pca = RandomForestClassifier(max_depth=None, min_samples_split=7, min_samples_leaf=2, n_estimators=10)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

## Apply Logistic Regression ##
from sklearn.linear_model import LogisticRegression
# clf_pca = LogisticRegression(C=2)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

## Apply Naive Bayes ##
from sklearn.naive_bayes import GaussianNB
# clf_pca = GaussianNB(priors=None)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

## Apply KNN ##
from sklearn.neighbors import KNeighborsClassifier
# clf_pca = KNeighborsClassifier(n_neighbors=9)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

## Apply Ordinary least squares ##
from sklearn import linear_model
# clf_pca = linear_model.LinearRegression()
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

# Apply Ridge Regression #
# clf_pca = linear_model.Ridge(alpha=.5, random_state=42)
# clf_pca.fit(x_train_2, y_train)
# predictions_test = clf_pca.predict(x_test_2)

# Apply Lasso Regression #
clf_pca = linear_model.Lasso(alpha=0.1,random_state=42)
clf_pca.fit(x_train_2, y_train)
predictions_test = clf_pca.predict(x_test_2)


# Predictions for Ordinary Least Squares, Ridge, and Lasso Regression
class_names = [0, 1]
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)


# Scatter plot
y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Section for OLS, Ridge Regression, Lasso Regression
Z[Z <= 0.5] = 0
Z[Z > 0.5] = 1
Z = Z.astype(int).reshape(xx.shape)


Z = Z.reshape(xx.shape)
# Draw contour line
plt.contour(xx, yy, Z)
plt.title('SVM Decision Surface')
plt.axis('off')
plt.show()