import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder



# Data load
data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv')
data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

zone = np.where(np.array(data["location"]) == "United States")

x = data.iloc[np.amin(zone) + 1:np.amax(zone) + 1, 2].values
x = LabelEncoder().fit_transform(x)
y = data.iloc[np.amin(zone) + 1:np.amax(zone) + 1, 5].values

xTrain = np.array(x).reshape(-1, 1)
yTrain = np.array(y).reshape(-1, 1)

#Divide data
trainRate = 0.7
trainIndex = int(len(xTrain) * trainRate)
xTest = xTrain[trainIndex:]
yTest = yTrain[trainIndex:]

xTrain = xTrain[:trainIndex]
yTrain = yTrain[:trainIndex]

# Scale the data
# scaler = StandardScaler()
# xTrain = scaler.fit_transform(xTrain)
# yTrain = scaler.fit_transform(yTrain)

# Regression
regression = LinearRegression()
pFeatures = PolynomialFeatures(degree=2, include_bias=False)
xPoly = pFeatures.fit_transform(xTrain)
regression.fit(xPoly, yTrain)

# Error calculation
errors = []
averageError = 0

for i in range(len(xTest)):
    error = regression.predict(pFeatures.fit_transform(xTest))[i] - yTest[i]
    errors.append(abs(error))
    plt.plot(xTest[i], regression.predict(pFeatures.fit_transform(xTest))[i], marker='.', color='g')
averageError = (sum(errors)) / len(xTest)
print("Average error: " + str(averageError))

# Plot settings
plt.plot(x, y, marker='.', color='#3F3BFA', label="Data line")
plt.plot(xTrain, regression.predict(xPoly), marker='', color='r', label="Regression line")
plt.grid(color='#1f1f1f', linestyle='dashed', linewidth=0.3)
plt.gca().set_facecolor('#ffffff')
plt.legend()
# plt.get_current_fig_manager().full_screen_toggle() #Shows the graph in fullscreen
plt.show()
