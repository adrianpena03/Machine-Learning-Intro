import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Load and prepare the data
df = pd.read_csv('diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Define parameters for testing
activation_functions = ['relu', 'identity', 'logistic', 'tanh']
hidden_layer_sizes = [(10,), (20,), (10, 10), (20, 20)]


results = {}
# Loop through each activation function
for func in activation_functions:
    # for each function, create an empty list in the dictionary
    # store the results for that specific function
    results[func] = []


# loop through each combination of activation function and hidden layer size
for activation in activation_functions:
    for size in hidden_layer_sizes:
        # Initialize the MLP classifier
        clf = MLPClassifier(
            random_state=1,
            hidden_layer_sizes=size,
            activation=activation,
            solver='adam',
            max_iter=1000,
            learning_rate='adaptive',
            alpha=0.00001,
            tol=0.0001,
            learning_rate_init=0.001
        )

        # Cross-validation for each metric
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        f1_score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()

        # Store the results
        results[activation].append({
            'hidden_layer_sizes': size,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })

# Plot overall accuracy vs hidden layer sizes for each activation function
plt.figure(figsize=(12, 8))
hidden_layer_labels = ['(10,)', '(20,)', '(10, 10)', '(20, 20)']
for activation in activation_functions:
    accuracies = [result['accuracy'] for result in results[activation]]
    plt.plot(hidden_layer_labels, accuracies, marker='o', label=activation)

# Customize the plot
plt.title('Overall Accuracy vs Hidden Layer Sizes for Different Activation Functions (10-fold Cross-Validation)')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Overall Accuracy')
plt.legend(title='Activation Function')
plt.grid(True)
plt.show()


for activation in activation_functions:
    print("Results for activation function:", activation)    
    # Go through each result for this activation function
    for result in results[activation]:
        print("Hidden Layer Sizes:", result['hidden_layer_sizes'], 
              "| Accuracy:", round(result['accuracy'], 4),
              "| Precision:", round(result['precision'], 4),
              "| Recall:", round(result['recall'], 4),
              "| F1 Score:", round(result['f1_score'], 4))
    print("\n")
