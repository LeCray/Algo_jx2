import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import sklearn
from numpy import loadtxt

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
        print(pred)

        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


#Initialize the model
model = Net()

model.load_state_dict(torch.load('./model.pth'))
model.eval()


def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()


ans = predict(np.array([[1,2,3,4]]))
print("Prediction: ", ans)
'''
example = np.array([[1.14724,1.14539,29.21052631578781,1094]])
example = torch.from_numpy(example).type(torch.FloatTensor)
traced_script_module = torch.jit.trace(model, example)

print("Tracing model...")
traced_script_module.save("traced_pytorch_model.pt")
print("Model successfully traced")


example2 = np.array([[1,3,4,5]])
output = traced_script_module(torch.from_numpy(example2).type(torch.FloatTensor))
print(output)
'''
