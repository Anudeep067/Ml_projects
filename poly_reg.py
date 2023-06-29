# Building the pol reg to predict the previous years sallery
# here data is less and we want to predict , so we not splitting data into training and test set
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:-1].values  # excluding first column
Y=dataset.iloc[:,-1].values
print(X)
print("-------------")
#Training the linear regression on the model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Training the polynomial Regression model on the whole dataset
# here we create a matrix containing x1,x1^2,x1^3,....
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)



# Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')     # allows to plot on 2d plane
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# visualing the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Predicting a new result with Lineare Regression
print(lin_reg.predict([[6.5]]))
print("............")
# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))