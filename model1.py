import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split


# Read the data from the train.txt file
data = []
labels = []

with open('D:/Downloads/nCRNA_s.train.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split(' ')
    label = int(parts[0])
    features = {}

    for feature in parts[1:]:
        index, value = feature.split(':')
        features[int(index)] = float(value)

    labels.append(label)
    data.append(features)

# Create a DataFrame from the data
df = pd.DataFrame(data)
df.fillna(0.0, inplace=True)  # Fill missing values with 0.0

# Convert the labels to a numpy array
y = pd.Series(labels).to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
from sklearn.impute import SimpleImputer

# Create an imputer to fill in missing values
imputer = SimpleImputer(strategy='mean')

# Preprocess the training data to fill in missing values
X_train = imputer.fit_transform(X_train)

# Preprocess the test data using the same imputer
X_test = imputer.transform(X_test)



# Create an SVM classifier and fit the training data
classifier = svm.SVC()
classifier.fit(X_train, y_train)
pickle.dump(classifier , open('model1.pkl' , 'wb'))