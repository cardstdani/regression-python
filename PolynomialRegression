import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Data load
data = pd.read_csv('My_Data.csv')
x = data.iloc[:, 1].values
y = data.iloc[:, 2].values

xTrain = np.array(x).reshape(-1, 1)
yTrain = np.array(y).reshape(-1, 1)

# Regression
regression = LinearRegression()
pFeatures = PolynomialFeatures(degree=1, include_bias=False)
xPoly = pFeatures.fit_transform(xTrain)
regression.fit(xPoly, yTrain)

# Error calculation
errors = []
averageError = 0

for i in range(len(xTrain)):
    error = regression.predict(xPoly)[i] - yTrain[i]
    errors.append(abs(error))
    plt.plot([xTrain[i], xTrain[i]], [regression.predict(xPoly)[i], regression.predict(xPoly)[i] - error], marker='',
             color='g')
averageError = (sum(errors)**2) / len(xTrain)
print("Average error: " + str(averageError))

# Plot settings
plt.plot(x, y, marker='.', color='#3F3BFA', label="Data line")
plt.plot(xTrain, regression.predict(xPoly), marker='', color='r', label="Regression line")
xTest = np.array([10]).reshape(-1, 1)
plt.plot(xTest, regression.predict(pFeatures.fit_transform(xTest)), marker='o', color='y', label="Prediction point")
plt.grid(color='#1f1f1f', linestyle='dashed', linewidth=0.3)
plt.gca().set_facecolor('#ffffff')
plt.legend()
# plt.get_current_fig_manager().full_screen_toggle() #Shows the graph in fullscreen
plt.show()
