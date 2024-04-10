# The source code with all results can also be found here: https://www.kaggle.com/code/tleonhardt/csc-428-lab-6

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('/kaggle/input/kdd-data/KDD-Data.csv')

# Drop any rows with missing values
data.dropna(inplace=True)

# Remove single quotes from column names
data.columns = data.columns.str.replace("'", "")

le = LabelEncoder()
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])
data['class'] = le.fit_transform(data['class'])

# Split features and target variable
y = data['class']
X = data.drop('class', axis=1)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.30, shuffle=True) 

#Standardize features: Transform the features
#such that they have a mean of 0 and a standard deviation of 1

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train logistic regression model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 10000000000000)
model.fit(X_train, y_train)

# Predict on the testing set and calculate accuracy

y_pred = model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(logistic_accuracy * 100))

# Initialize and train KNN model

knn = KNeighborsClassifier(n_neighbors=5) 
  
knn.fit(X_train, y_train) 

# Predict on the testing set and calculate accuracy
y_pred = knn.predict(X_test)
# Calculate accuracy
knn_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(knn_accuracy * 100))


print("Logistic Regression Accuracy: {:.2f}%".format(logistic_accuracy * 100))
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))