import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

class SensorNet(nn.Module):
    def __init__(self):
        super(SensorNet, self).__init__()
        self.fc1 = nn.Linear(29 * 3, 256)  # 输入 29x3, 输出 256
        self.fc2 = nn.Linear(256, 256)     # 输出 256
        self.fc3 = nn.Linear(256, 1)       # 输出 y1
        self.dropout = nn.Dropout(0.5)     # Dropout 防止过拟合

    def forward(self, x):
        x = F.relu(self.fc1(x))
        sensor_feat = F.relu(self.fc2(x))  # 提取中间层特征 256x1
        sensor_feat = self.dropout(sensor_feat)  # 在中间层使用 Dropout
        y1 = self.fc3(sensor_feat)
        #y1= torch.relu(self.fc3(sensor_feat))
        #y1 = F.leaky_relu(self.fc3(sensor_feat))
        #y1 = torch.sigmoid(self.fc3(sensor_feat))
        #y1 = F.softplus(self.fc3(sensor_feat))

        return sensor_feat, y1  # 返回中间特征和最终输出

class ElectricNet(nn.Module):
    def __init__(self):
        super(ElectricNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(256 + 128, 64)  # 拼接传感器特征和LSTM特征后，降维到64
        self.fc2 = nn.Linear(64, 1)
        #
        self.dropout = nn.Dropout(0.5)

    def forward(self, sensor_feat, electric_data):
        lstm_out, (hn, cn) = self.lstm(electric_data)
        lstm_feat = torch.mean(lstm_out, dim=1)  # 对所有时间步的特征取均值
        combined = torch.cat((sensor_feat, lstm_feat), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        #y2 = F.leaky_relu(self.fc2(x))
        #y2 = F.softplus(self.fc2(x))
        y2 = self.fc2(x)
        return y2
