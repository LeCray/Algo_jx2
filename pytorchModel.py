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

X = trainingData[0,0:4]
#print("Numpy: ", X)
y = trainingData[:,4]
A = testingData[:,0:4]
b = testingData[:,4]

X = torch.from_numpy(X).type(torch.FloatTensor)
#print("Tensor: ", X)
y = torch.from_numpy(y).type(torch.LongTensor)

A = torch.from_numpy(A).type(torch.FloatTensor)
b = torch.from_numpy(b).type(torch.LongTensor)

#our class must extend nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12,4)
        self.fc3 = nn.Linear(4,2)

    #This must be implemented
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    #This function takes an input and predicts the class, (0 or 1)
    def predict(self,x):
        #Apply softmax to output
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


#Initialize the model
model = Net()
#Define loss criterion
criterion = nn.CrossEntropyLoss()
#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Number of epochs
epochs = 5000 #This was 50,000
#List to store losses
losses = []
for i in range(epochs):
    #Precit the output for Given input
    y_pred = model.forward(X)
    #Compute Cross entropy loss
    loss = criterion(y_pred,y)
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

    print("Step:",i, "| Loss:", loss.item())


from sklearn.metrics import accuracy_score
print("Accuracy Score: ", accuracy_score(model.predict(A),b))


def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()
