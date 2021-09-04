import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
import sklearn

# Data load
df = pd.read_csv(
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv")
df = df.loc[df['location'] == 'World']

x = np.array(range(0, len(df['total_vaccinations'])))
y = np.array(df['total_vaccinations'])

xTrain = np.array(x).reshape(-1, 1)
yTrain = np.array(y).reshape(-1, 1)

# Split data
trainRate = 0.7
trainIndex = int(len(xTrain) * trainRate)
xTest = xTrain[trainIndex:]
yTest = yTrain[trainIndex:]

xTrain = xTrain[:trainIndex]
yTrain = yTrain[:trainIndex]

# Scale data
# scaler = StandardScaler()
# xTrain = scaler.fit_transform(xTrain)
# yTrain = scaler.fit_transform(yTrain)

# Regression
regression = LinearRegression()
pFeatures = PolynomialFeatures(degree=2, include_bias=False)
xPoly = pFeatures.fit_transform(xTrain)
regression.fit(xPoly, yTrain)

# Error calculation
print("Average error: " + str(sklearn.metrics.mean_absolute_error(yTest, regression.predict(pFeatures.fit_transform(xTest)))))
print("Squared error: " + str(sklearn.metrics.mean_squared_error(yTest, regression.predict(pFeatures.fit_transform(xTest)))))
"""errors = []
squaredErrors = []

for i in range(len(xTest)):
    error = regression.predict(pFeatures.fit_transform(xTest))[i] - yTest[i]
    errors.append(abs(error))
    squaredErrors.append(abs(error) ** 2)
mae = '%f' % (sum(errors) / len(xTest))
mse = '%f' % (sum(squaredErrors) / len(xTest))
print("Average error: " + str(mae))
print("Squared error: " + str(mse))"""

# Plot settings
plt.plot(x, y, marker='.', color='#3F3BFA', label="Data line")
plt.plot(xTrain, regression.predict(xPoly), marker='', color='r', label="Train regression line")
plt.plot(xTest, regression.predict(pFeatures.fit_transform(xTest)), marker='', color='g', label="Test regression line")
plt.grid(color='#1f1f1f', linestyle='dashed', linewidth=0.3)
plt.gca().set_facecolor('#ffffff')
plt.legend()
plt.show()
