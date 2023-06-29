import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('meter3.csv')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, -1].values
print(X)
print(".....")
print(y)
## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)
print(X_train)
print(X_test)

from sklearn.linear_model import LinearRegression
X_train1= X_train.reshape(-1, 1)
X_test1 = X_test.reshape(-1, 1)
y_train1= y_train.reshape(-1, 1)
y_test1 = y_test.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X_train1, y_train1)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
y_pred = regressor.predict(X_test1)
print("..........")
print(y_pred)
#arr = np.array(X_train1)
# convert to tuple
#tup = tuple(arr)
# set tuple as key
print("............................")
#    print(tup)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train1), color = 'blue')
plt.title('power vs voltage(Training set)')
plt.xlabel('voltage')
plt.ylabel('power')
plt.show()
## Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train1), color = 'blue')
plt.title('power vs voltage (Test set)')
plt.xlabel('voltage')
plt.ylabel('power')
plt.show()
v = X_test.astype(np.float64)
p=y_test.astype(np.float64)


z
regressor.fit(X_test1, I1)
plt.scatter(X_test, I, color = 'red')
plt.plot(X_test, regressor.predict(X_test1), color = 'yellow')
plt.title('current vs voltage(test set)')
plt.xlabel('voltage')
plt.ylabel('current')
plt.show()
## polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_test1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_test1)
#visualising polynomialreg
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, lin_reg_2.predict(poly_reg.fit_transform(X_test1)), color = 'blue')
plt.title('power vs volt (Polynomial Regression)')
plt.xlabel('volt')
plt.ylabel('power')
plt.show()
#polregforiv
lin_reg_2.fit(X_poly, I1)

plt.scatter(X_test, I, color = 'red')
plt.plot(X_test, lin_reg_2.predict(poly_reg.fit_transform(X_test1)), color = 'blue')
plt.title('current vs volt (Polynomial Regression)')
plt.xlabel('volt')
plt.ylabel('current')
plt.show()
#svm

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_I = StandardScaler()
Xsvr = sc_X.fit_transform(X_test1)
Isvr= sc_I.fit_transform(I1)

## Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(Xsvr, Isvr)

## Predicting a new result
sc_I.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)) # 6.5 may be any number for predicton

## Visualising the SVR results
plt.scatter(sc_X.inverse_transform(Xsvr), sc_I.inverse_transform(Isvr), color = 'red')
plt.plot(sc_X.inverse_transform(Xsvr), sc_I.inverse_transform(regressor.predict(Xsvr).reshape(-1,1)), color = 'blue')
plt.title('current vs volt(SVR)')
plt.xlabel('voltage')
plt.ylabel('current')
plt.show()

## Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(Xsvr)), max(sc_X.inverse_transform(Xsvr)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(Xsvr), sc_I.inverse_transform(Isvr), color = 'red')
plt.plot(X_grid, sc_I.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('current vs volt (SVR)')
plt.xlabel('volt')
plt.ylabel('current')
plt.show()