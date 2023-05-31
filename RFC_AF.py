from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with open('AFRICA_CL.data', 'r') as f:
    s = f.read()
with open('AFRICA_CL.data', 'w') as f:
    s = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(s)

dataCL = pd.read_csv('AFRICA_CL.data')
dataCL = pd.DataFrame(dataCL)
dataCL.drop('year', axis=1)
dataCL.insert(loc=0, column='row_num', value=np.arange(len(dataCL)))

meansTC = dataCL['VCI']

droughtLevel = []
for x in meansTC:
    if 0 <= x < 6:
        droughtLevel.append(5)
    elif 6 <= x < 16:
        droughtLevel.append(4)
    elif 16 <= x < 26:
        droughtLevel.append(3)
    elif 26 <= x < 35:
        droughtLevel.append(2)
    elif 35 <= x <= 40:
        droughtLevel.append(1)
    elif x > 40:
        droughtLevel.append(0)

df = dataCL.drop('VCI', axis=1)

X = df
print(X)

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])

print(X.head())
print(y.head())

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

m_depth = np.arange(1, 32)
train_accuracy = np.empty(len(m_depth))
test_accuracy = np.empty(len(m_depth))

# Loop over K values
for i, k in enumerate(m_depth):
    clf = RandomForestClassifier(n_estimators=51, max_depth=k)
    clf.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = clf.score(X_train, y_train)
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.plot(m_depth, test_accuracy, label='Testing dataset Accuracy')
plt.plot(m_depth, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()

#
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters
# clf.fit(X_train, y_train)
#
# # performing predictions on the test dataset
# y_pred = clf.predict(X_test)
#
# print()
#
# # using metrics module for accuracy calculation
# print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
#
# print(clf.predict([[10.5, 15, 30, 18]]))
#
# #feature_imp = pd.Series(clf.feature_importances_,).sort_values(ascending = False)
# #feature_imp
