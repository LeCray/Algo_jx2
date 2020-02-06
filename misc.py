import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import sklearn
from numpy import loadtxt

dataset = loadtxt('Data/BuyData.csv', delimiter=',') #10357

row_count = sum(1 for row in dataset)

trainingData = dataset[0:round(row_count*0.7),:]
testingData = dataset[round(row_count*0.7):row_count,:]

X = trainingData[:,0:4]
y = trainingData[:,4]

count = 0

for i in y:
    if i == 1:
        count = count + 1

print("Count:",count)
print("Row Count:",row_count)
print("Ratio:", (count/row_count)*100)
