import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_root = "https://raw.githubusercontent.com/ageron/data/main"
lifesat = pd.read_csv(data_root + "/lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(x="GDP per capita (USD)", y="Life satisfaction", kind="scatter", grid=True)
plt.axis([23_500, 65_300, 5.0, 8.0])
plt.savefig("plot.png")

model = LinearRegression() # Create linear regression model using least squares: calculate square error from model and minimize it
# can also use other estimators, such as ridge, lasso?
model.fit(X, y)

X_new = [[89688.9569584859]] # Ireland
print(model.predict(X_new))