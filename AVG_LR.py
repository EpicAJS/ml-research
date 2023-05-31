import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt

data2019x = pd.read_csv('drought2019.data')
data2019t = pd.DataFrame(data2019x)
data2019w = data2019t.drop('year', axis=1)
data2019 = data2019w.drop('week', axis=1)
print(data2019.head())
data2019['means'] = data2019.iloc[:, 0:20].mean(axis=1)
print(data2019.head())


data2020x = pd.read_csv('drought2020.data')
data2020t = pd.DataFrame(data2020x)
data2020w = data2020t.drop('year', axis=1)
data2020 = data2020w.drop('week', axis=1)
print(data2020.head())
data2020['means'] = data2020.iloc[:, 0:20].mean(axis=1)
print(data2020.head())

# Features / Labels
X = data2019['means'].values.flatten().reshape(-1, 1)
y = data2020['means'].values.flatten().reshape(-1, 1)
print(data2020.info())

# algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

plt.scatter(X_train, y_train, label='Training Data', color='r')
plt.scatter(X_test, y_test, label='Testing Data', color='g')
plt.legend()
plt.title('Test Train Split')
plt.show()

# train
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions: ", predictions)
print("R^2 Vale: ", l_reg.score(X, y))
print("coedd: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)

plt.plot(X_test, predictions, label='Linear Regression', color='b');
# plt.plot(np.unique(X), np.poly1d(np.polyfit(X, y, 1))(np.unique(X)))

plt.scatter(X_test, y_test, label='Actual Testing Data', color='g')

plt.legend()
plt.show()
