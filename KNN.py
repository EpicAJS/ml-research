from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with open('USA_VHI_CL.data', 'r') as f:
    s = f.read()
with open('USA_VHI_CL.data', 'w') as f:
    s = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(s)

with open('USA_TCI_CL.txt', 'r') as f:
    s = f.read()
with open('USA_TCI_CL.txt', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

with open('USA_VCI_CL.data', 'r') as f:
    s = f.read()
with open('USA_VCI_CL.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

with open('USA_VHI_CL.data', 'r') as f:
    s = f.read()
with open('USA_VHI_CL.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)
import pandas as pd
import numpy as np

dataVC = pd.read_csv('USA_VCI_CL.data')
dataVC = pd.DataFrame(dataVC)
dataVC = dataVC.drop('year', axis=1)
dataVC = dataVC.drop('week', axis=1)

dataVC.to_numpy()
print(dataVC)

meanVC = []
avgVC = 0
for col, row in dataVC.iterrows():
    print(f"Index: {col}")
    avgVC = ((float(f"{row['vh1']}")/100) * 1) + ((float(f"{row['vh2']}")/100) * 6) + \
            ((float(f"{row['vh3']}")/100) * 11) + ((float(f"{row['vh4']}")/100) * 16) + \
            ((float(f"{row['vh5']}")/100) * 21) + ((float(f"{row['vh6']}")/100) * 26) + \
            ((float(f"{row['vh7']}")/100) * 31) + ((float(f"{row['vh8']}")/100) * 36) + \
            ((float(f"{row['vh9']}")/100) * 41) + ((float(f"{row['vh10']}")/100) * 46) + \
            ((float(f"{row['vh11']}")/100) * 51) + ((float(f"{row['vh12']}")/100) * 56) + \
            ((float(f"{row['vh13']}")/100) * 61) + ((float(f"{row['vh14']}")/100) * 66) + \
            ((float(f"{row['vh15']}")/100) * 71) + ((float(f"{row['vh16']}")/100) * 76) + \
            ((float(f"{row['vh17']}")/100) * 81) + ((float(f"{row['vh18']}")/100) * 86) + \
            ((float(f"{row['vh19']}")/100) * 91) + ((float(f"{row['vh20']}")/100) * 96) + \
            ((float(f"{row['vh21']}")/100) * 101)
    print(avgVC)
    meanVC.append(avgVC)
print(meanVC)

dataTC = pd.read_csv('USA_TCI_CL.txt')
dataTC = pd.DataFrame(dataTC)
dataTC = dataTC.drop('year', axis=1)
dataTC = dataTC.drop('week', axis=1)

dataTC.to_numpy()
print(dataTC)

meanTC = []
avgTC = 0
for col, row in dataTC.iterrows():
    print(f"Index: {col}")
    avgTC = ((float(f"{row['vh1']}")/100) * 1) + ((float(f"{row['vh2']}")/100) * 6) + \
            ((float(f"{row['vh3']}")/100) * 11) + ((float(f"{row['vh4']}")/100) * 16) + \
            ((float(f"{row['vh5']}")/100) * 21) + ((float(f"{row['vh6']}")/100) * 26) + \
            ((float(f"{row['vh7']}")/100) * 31) + ((float(f"{row['vh8']}")/100) * 36) + \
            ((float(f"{row['vh9']}")/100) * 41) + ((float(f"{row['vh10']}")/100) * 46) + \
            ((float(f"{row['vh11']}")/100) * 51) + ((float(f"{row['vh12']}")/100) * 56) + \
            ((float(f"{row['vh13']}")/100) * 61) + ((float(f"{row['vh14']}")/100) * 66) + \
            ((float(f"{row['vh15']}")/100) * 71) + ((float(f"{row['vh16']}")/100) * 76) + \
            ((float(f"{row['vh17']}")/100) * 81) + ((float(f"{row['vh18']}")/100) * 86) + \
            ((float(f"{row['vh19']}")/100) * 91) + ((float(f"{row['vh20']}")/100) * 96) + \
            ((float(f"{row['vh21']}")/100) * 101)
    print(avgTC)
    meanTC.append(avgTC)
print(meanTC)

conditions = [meanVC, meanTC]
conditions = np.array(conditions).reshape(2057, 2)

df = pd.DataFrame(conditions, columns=['meansVC', 'meansTC'])

print(df)
df['indexAvg'] = df.iloc[:, 0:1].mean(axis=1)


dataVH = pd.read_csv('USA_VHI_CL.data')
dataVH = pd.DataFrame(dataVH)
dataVH = dataVH.drop('weekVH', axis=1)
dataVH = dataVH.drop('year', axis=1)

dataVH.to_numpy()
print(dataVH)

meanVH = []
avgVH = 0
for col, row in dataVH.iterrows():
    print(f"Index: {col}")
    avgVH = ((float(f"{row['vh1']}")/100) * 1) + ((float(f"{row['vh2']}")/100) * 6) + \
            ((float(f"{row['vh3']}")/100) * 11) + ((float(f"{row['vh4']}")/100) * 16) + \
            ((float(f"{row['vh5']}")/100) * 21) + ((float(f"{row['vh6']}")/100) * 26) + \
            ((float(f"{row['vh7']}")/100) * 31) + ((float(f"{row['vh8']}")/100) * 36) + \
            ((float(f"{row['vh9']}")/100) * 41) + ((float(f"{row['vh10']}")/100) * 46) + \
            ((float(f"{row['vh11']}")/100) * 51) + ((float(f"{row['vh12']}")/100) * 56) + \
            ((float(f"{row['vh13']}")/100) * 61) + ((float(f"{row['vh14']}")/100) * 66) + \
            ((float(f"{row['vh15']}")/100) * 71) + ((float(f"{row['vh16']}")/100) * 76) + \
            ((float(f"{row['vh17']}")/100) * 81) + ((float(f"{row['vh18']}")/100) * 86) + \
            ((float(f"{row['vh19']}")/100) * 91) + ((float(f"{row['vh20']}")/100) * 96) + \
            ((float(f"{row['vh21']}")/100) * 101)
    print(avgVH)
    meanVH.append(avgVH)
print(meanVH)


droughtLevel = []
for x in meanVH:
    if 0 <= x < 6:
        droughtLevel.append("Exceptional")
    if 6 <= x < 16:
        droughtLevel.append("Extreme")
    if 16 <= x < 26:
        droughtLevel.append("Severe")
    if 26 <= x < 35:
        droughtLevel.append("Moderate")
    if 35 <= x <= 40:
        droughtLevel.append("Abnormally Dry")
    if x > 40:
        droughtLevel.append("No Drought")

X = df

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])

print(X.head())
print(y.head())

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.xticks(range(0, 21))

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
