import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt

vhi13 = pd.read_csv('vhi13.data')
vhi13 = pd.DataFrame(vhi13)
vhi13 = vhi13.drop('year', axis=1)
vhi13 = vhi13.drop('week', axis=1)
print(vhi13.head())
# vhi13['means'] = vhi13.iloc[:, 0:20].mean(axis=1)
print(vhi13.head())

tci13 = pd.read_csv('tci13.data')
tci13 = pd.DataFrame(tci13)
tci13 = tci13.drop('year', axis=1)
tci13 = tci13.drop('week', axis=1)
print(tci13.head())
# tci13['means'] = tci13.iloc[:, 0:20].mean(axis=1)
print(tci13.head())

# Features / Labels
# X = vhi13['means'].values.flatten().reshape(-1, 1)
# y = tci13['means'].values.flatten().reshape(-1, 1)

X = vhi13.values.flatten().reshape(-1, 1)
y = tci13.values.flatten().reshape(-1, 1)

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
