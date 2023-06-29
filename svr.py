# Support vector regression
# invented by vladmir vapnik in 90's bell labs
# in svr we have a tube of radius epsilon where epsilon is the margin of error
import numpy as np           # importing libraries
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv') # importing datasheet
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print("...........")
print(y)
print("........")
y = y.reshape(len(y),1)       # arguments are rows, columns
print("..........")
print(y)
from sklearn.preprocessing import StandardScaler   # feature scaling
sc_X = StandardScaler()                            # here we wont split data into training and data set
sc_y = StandardScaler()                #  we want to extract exact correlation from the data
X = sc_X.fit_transform(X)              # we wont apply feature scaling for dummy vairable
y = sc_y.fit_transform(y)

X1 =X.reshape(-1, 1)
y1 =y.reshape(-1, 1)


# when dependent values takes binary values we dont apply feature scaling
print("......")           # when dependent values take super high values with re[ect to others we apply feature scaling
print(X)
print("......")
print(y)
from sklearn.svm import SVR   # training svr model on whole dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X1, y1)

# predicting the new result

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# visualising the svr results
plt.scatter(sc_X.inverse_transform(X1), sc_y.inverse_transform(y1), color = 'red')
plt.plot(sc_X.inverse_transform(X1), sc_y.inverse_transform(regressor.predict(X1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# visualising the svr for higher resolution and smoother
X_grid = np.arange(min(sc_X.inverse_transform(X1)), max(sc_X.inverse_transform(X1)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y1), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()