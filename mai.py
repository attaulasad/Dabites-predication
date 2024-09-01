import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset into a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the dataset
print(diabetes_dataset.head())

# Print the number of rows and columns in the dataset
print("Shape of the dataset:", diabetes_dataset.shape)

# Get the statistical measures of the data
print("Statistical measures:")
print(diabetes_dataset.describe())

# Check the distribution of the target variable (Outcome)
print("Outcome distribution:")
print(diabetes_dataset['Outcome'].value_counts())

# Calculate and print the mean of each feature, grouped by Outcome
print("Mean of each feature grouped by Outcome:")
print(diabetes_dataset.groupby('Outcome').mean())

# Separate the features and the target (Outcome)
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the feature data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Print the shapes of the original, training, and testing data
print("Original data shape:", X.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize the Support Vector Machine classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Train the SVM classifier on the training data
classifier.fit(X_train, Y_train)

# Make predictions on the test data
X_test_prediction = classifier.predict(X_test)

# Calculate and print the accuracy score on the test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

# Predicting for a new input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we're predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

# Make a prediction for the new data
prediction = classifier.predict(std_data)

# Print the prediction result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
