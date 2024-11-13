import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('diabetes.csv', index_col=0)

# Standardize the Features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the data into Train and Test sets
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# hyperparameter values for C
C_values = [0.5, 1.0, 1.5, 2.0]

# kernels to test
kernel_types = ['poly', 'linear', 'sigmoid', 'rbf']

# dictionary to store metrics for different kernels
kernel_metrics = {
    'accuracy': {
        'poly': [],
        'linear': [],
        'sigmoid': [],
        'rbf': []
    },
    'precision': {
        'poly': [],
        'linear': [],
        'sigmoid': [],
        'rbf': []
    },
    'recall': {
        'poly': [],
        'linear': [],
        'sigmoid': [],
        'rbf': []
    },
    'f1score': {
        'poly': [],
        'linear': [],
        'sigmoid': [],
        'rbf': []
    },
    'roc_auc': {
        'poly': [],
        'linear': [],
        'sigmoid': [],
        'rbf': []
    }
}

# iterate over kernel types
for kernel in kernel_types:
    if kernel == 'poly':
        degree_values = [2, 3, 4]  # degrees to test for polynomial kernel
        for degree in degree_values:
            for C in C_values:
                print(f'Kernel: {kernel}, Degree: {degree}, C: {C}')
                clf = SVC(C=C, kernel=kernel, degree=degree, random_state=42)
                
                # Accuracy
                accuracy = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy').mean()
                kernel_metrics['accuracy'][kernel].append((C, degree, accuracy))
                
                # Precision
                precision = cross_val_score(clf, x_train, y_train, cv=10, scoring='precision').mean()
                kernel_metrics['precision'][kernel].append((C, degree, precision))
                
                # Recall
                recall = cross_val_score(clf, x_train, y_train, cv=10, scoring='recall').mean()
                kernel_metrics['recall'][kernel].append((C, degree, recall))
                
                # F1 Score
                f1score = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1').mean()
                kernel_metrics['f1score'][kernel].append((C, degree, f1score))

                # Print overall results for each kernel/degree/C
                print(f'Overall Results - Kernel: {kernel}, Degree: {degree}, C: {C}')
                print(f'Accuracy: {accuracy}')
                print(f'Precision: {precision}')
                print(f'Recall: {recall}')
                print(f'F1 Score: {f1score}')
                print('-' * 40)

    else:  # For other kernels (linear, sigmoid, rbf), no degree parameter
        for C in C_values:
            print(f'Kernel: {kernel}, C: {C}')
            clf = SVC(C=C, kernel=kernel, random_state=42)
            
            # Accuracy
            accuracy = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy').mean()
            kernel_metrics['accuracy'][kernel].append((C, None, accuracy))
            
            # Precision
            precision = cross_val_score(clf, x_train, y_train, cv=10, scoring='precision').mean()
            kernel_metrics['precision'][kernel].append((C, None, precision))
            
            # Recall
            recall = cross_val_score(clf, x_train, y_train, cv=10, scoring='recall').mean()
            kernel_metrics['recall'][kernel].append((C, None, recall))
            
            # F1 Score
            f1score = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1').mean()
            kernel_metrics['f1score'][kernel].append((C, None, f1score))

            # Print overall results for each kernel/C
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1score}')
            print('-' * 40)

# plotting the results
plt.figure(figsize=(12, 8))

# plot poly kernel results (for each degree, make separate lines)
for degree in [2, 3, 4]:
    poly_accuracies = [accuracy for (C, deg, accuracy) in kernel_metrics['accuracy']['poly'] if deg == degree]
    plt.plot(C_values, poly_accuracies, linestyle='dashed', marker='o', label=f'Poly (degree={degree})')

# plot linear, sigmoid, and rbf kernel results for accuracy
for kernel in ['linear', 'sigmoid', 'rbf']:
    accuracies = [accuracy for (C, _, accuracy) in kernel_metrics['accuracy'][kernel]]
    plt.plot(C_values, accuracies, linestyle='dashed', marker='o', label=f'{kernel.capitalize()} Kernel')

# add titles and labels for the accuracy plot
plt.title('Accuracy vs C_value for different SVM Kernels')
plt.xlabel('C-value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()