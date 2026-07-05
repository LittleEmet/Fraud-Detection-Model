import torch
import joblib
import torch.nn as nn
from preprocess import clean, encode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('bank_fraud.csv')
df=clean(df)
X,y=encode(df)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler,'scaler.pkl')
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
from torch.utils.data import TensorDataset, DataLoader
dataset=TensorDataset(X_train,y_train)
batch_size=512
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
from Model import Net
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    running_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*batch_X.size(0)
    epoch_loss= running_loss/len(dataloader.dataset)
    if epoch % 10 == 0:
        print(epoch_loss)
torch.save(model.state_dict(),'model.pth')