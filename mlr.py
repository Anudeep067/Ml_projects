import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print("...................................")
# encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("......................")
print(X_train)
print("......................")
# Training the multiple linear Regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
# here we going to disply to 10 (20%)coloums one with real profits and other with predicted values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)     # to print to decimals
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
 # concaticates to vectors vertically or horizontally
 # reshape converts horizontzl to vertical rows = y_pred, columns =1
 # in concatenate fun 0 argument  means vertical concatination and 1 means horizontal concatination
plt.show()
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.cla()