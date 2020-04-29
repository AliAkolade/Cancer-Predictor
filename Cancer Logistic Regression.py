# Importing the libraries
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 23].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Logistic Regression
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

# Making the Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Show Accuracy
result = logreg.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * result))

# Save Model to File
pickle.dump(logreg, open('LogReg.pk1', 'wb'))
