import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
dataset = pd.read_csv('meter3.csv')
X = dataset.iloc[2:, 0].values
y = dataset.iloc[2:, -1].values

print(X)
print("....sdgfdg.")
print(y)
X=X.astype(np.float64)
y=y.astype(np.float64)
print(X)

