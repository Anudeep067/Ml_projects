import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, 2].values

X= X.reshape(-1, 1)


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
regressor.predict([[6.5]])
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
print(X_grid)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Actual power vs reactive energy (Random Forest Regression)')
plt.xlabel('Actual power')
plt.ylabel('Reactive energy')
plt.show()