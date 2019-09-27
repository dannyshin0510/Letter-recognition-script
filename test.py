import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()
clf = DecisionTreeClassifier()



#CIBC user interaction
print("How much would you like to eTransfer?")
input()
#The training procedure
xtrain = data[0:21000, 1:]
train_label=data[0:21000, 0]

clf.fit(xtrain, train_label)

#Testing data
xtest=data[21000:, 1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)
count = 0
for i in range(0, 21000):
    count+=1 if p[i]== actual_label[i] else 0
print ("The accuracy of the model's prediction is: ", (count/21000)*100)

#Sample test case for blackbox function tests.
# d = xtest[8]
# d.shape=(28,28)
# pt.imshow(255-d,cmap='gray')
# print(clf.predict([xtest[8]]))
# pt.show()
