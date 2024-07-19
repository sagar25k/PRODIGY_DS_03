# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import numpy as np
import urllib.request
import zipfile
import os

# Define the URL for the dataset and the local path to store it
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
local_zip_file = 'bank.zip'
local_folder = 'bank'

# Download the dataset if it doesn't exist
if not os.path.exists(local_zip_file):
    urllib.request.urlretrieve(url, local_zip_file)

with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
    zip_ref.extractall(local_folder)


data_file = os.path.join(local_folder, 'bank-full.csv')
df = pd.read_csv(data_file, sep=';')


print("First few rows of the dataset:")
print(df.head())


df = df.drop_duplicates()


df = df.fillna(method='ffill')  # Forward fill


df = df.astype({
    'age': 'int64',
    'balance': 'int64',
    'day': 'int64',
    'duration': 'int64',
    'campaign': 'int64',
    'pdays': 'int64',
    'previous': 'int64'
})


df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 4. Remove outliers
# Using the IQR method to remove outliers for numerical columns

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))
    return df.loc[filter]

numerical_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
for column in numerical_columns:
    df = remove_outliers(df, column)

# Verify data types and clean data
print("\nData types after cleaning:")
print(df.dtypes)
print("\nSummary statistics of cleaned data:")
print(df.describe())


X = df.drop(columns='y')
y = df['y']

# Preprocess categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the target for the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\nAccuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Optional: Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Plotting data distributions to verify cleaning
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Distribution of {column}')
    plt.show()
