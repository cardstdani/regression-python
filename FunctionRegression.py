import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder

x = []
y = []

dataLength = 100
testLength = int(dataLength * 0.5)
for i in range(dataLength):  
  x.append(i)
  #Function of x(i) here
  y.append(i*i + 2)

#Split dataset into train and test
x_train = tf.constant(x[:testLength])
y_train = tf.constant(y[:testLength])

x_test = tf.constant(x[testLength:])
y_test = tf.constant(y[testLength:])

#Make the regression
regression = LinearRegression()
pFeatures = PolynomialFeatures(degree=2, include_bias=False)
xPoly = pFeatures.fit_transform(tf.reshape(x_train, (-1, 1)))
history = regression.fit(xPoly, tf.reshape(y_train, (-1, 1)))

#Plot results
plt.plot(x_train, y_train, color='b')
plt.plot(x_train, regression.predict(xPoly), color='r')
plt.plot(x_test, regression.predict(pFeatures.fit_transform(tf.reshape(x_test, (-1, 1)))), color='g')
print(regression.predict(pFeatures.fit_transform(tf.reshape([3000], (-1, 1)))))
