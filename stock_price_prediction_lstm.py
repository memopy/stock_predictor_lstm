import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((-1,1))
train_df = yf.download("TUPRS.IS","2013-01-01","2023-01-01")
test_df = yf.download("TUPRS.IS","2023-01-01","2024-01-01")
train_list = scaler.fit_transform(train_df["Close"].values.tolist())
test_list = scaler.fit_transform(test_df["Close"].values.tolist())

class StockDataset(Dataset):
    def __init__(self,stock_chart,seq_len):
        super().__init__()
        self.stock_chart = stock_chart
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.stock_chart)-self.seq_len

    def __getitem__(self,index):
        x = self.stock_chart[index:self.seq_len+index]
        y = self.stock_chart[index+self.seq_len]
        return torch.tensor(x).to(torch.float),torch.tensor(y).to(torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = StockDataset(train_list,7)
train = DataLoader(train,100,True)

test = StockDataset(test_list,7)
test = DataLoader(test,1,False)

class StockPredictorLSTM(nn.Module):
    def __init__(self,input_size,output_size,layer_count):
        super().__init__()
        self.lstm = nn.LSTM(input_size,output_size,layer_count)
        self.fc = nn.Linear(output_size,1)
    
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.fc(x[:,-1])
        return x

stock_predictor = StockPredictorLSTM(1,16,1).to(device)

lr = 0.001
epochs = 500

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(stock_predictor.parameters(),lr)

running_loss = 0
for epoch in range(1,epochs+1):
    for prices,labels in train:
        prices,labels = prices.to(device),labels.to(device)
        loss = criterion(stock_predictor(prices),labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    running_loss += loss.item()
    print(f"{epoch}. EPOCH DONE. LOSS : {loss}. RUNNING LOSS : {running_loss/epoch}")

stock_predictor.eval()
predictions = []
for price,_ in test:
    price = price.to(device)
    predictions.append(stock_predictor(price).item())

plt.plot(test_df.index[7:], test_list[7:],color="r",label="Original")
plt.plot(test_df.index[7:], predictions,color="g",label="Predicted")
plt.legend()
plt.show()