import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



#Setting up data sets, scaling, and basic info
df = pd.read_csv('./Desktop/CSV/Cardinfo.csv')
X = df.iloc[:, 0: 7]
y = df.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=69)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
EPOCHS = 100000
Batch_Size = 49
Learning_Rate = 0.00000001

#Training data used
class CreditTrain(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self,index):
        return self.X_data[index], self.y_data[index]
    def __len__(self) :
        return len(self.X_data)
train_data = CreditTrain(torch.FloatTensor(X_train), torch.FloatTensor(y_train))

#Testing data
class CreditTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__(self):
        return len(self.X_data)
test_data = CreditTest(torch.FloatTensor(X_test))

#Initialize Data Loaders
train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

#Final Classification layer
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 7.
        self.layer_1 = nn.Linear(7, 49) 
        self.layer_2 = nn.Linear(49, 49)
        self.layer_out = nn.Linear(49, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(49)
        self.batchnorm2 = nn.BatchNorm1d(49)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


#Check to see if GPU is available for use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Intitializing the data optamizer
model = BinaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)

#function for calculating efficiency of data
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
#Training the Model
model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        #Print Results from each Epoch - Algorithm Run
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
y_pred_list = []
model.eval()
#Set Up Random Test Case
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
#
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
confusion_matrix(y_test, y_pred_list)
print(accuracy_score(y_test, y_pred_list))
