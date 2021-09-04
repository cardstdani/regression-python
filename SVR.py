import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
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

#Scale data
scalerX = StandardScaler()
scalerY = StandardScaler()
xTrain = scalerX.fit_transform(xTrain)
yTrain = scalerY.fit_transform(yTrain)

# Split data
trainRate = 0.9
trainIndex = int(len(xTrain) * trainRate)
xTest = xTrain[trainIndex:]
yTest = yTrain[trainIndex:]

xTrain = xTrain[:trainIndex]
yTrain = yTrain[:trainIndex]

# Regression
model = SVR(kernel="rbf")
model.fit(xTrain, yTrain)

# Error calculation
print("Average error: " + str(sklearn.metrics.mean_absolute_error(yTest, model.predict(xTest))))
print("Squared error: " + str(sklearn.metrics.mean_squared_error(yTest, model.predict(xTest))))
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
plt.plot(scalerX.transform(x.reshape(-1, 1)), scalerY.transform(y.reshape(-1, 1)), marker='.', color='#3F3BFA', label="Data line")
plt.plot(xTrain, model.predict(xTrain), marker='', color='r', label="Train regression line")
plt.plot(xTest, model.predict(xTest), marker='', color='g', label="Test regression line")
plt.grid(color='#1f1f1f', linestyle='dashed', linewidth=0.3)
plt.gca().set_facecolor('#ffffff')
plt.legend()
plt.show()
