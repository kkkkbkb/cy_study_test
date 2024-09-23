import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def custom_loss(y2, y1, y_real):
    criterion = nn.SmoothL1Loss()  # Huber 损失
    loss1 = criterion(y2, y1)
    loss2 = criterion(y_real, y1)
    return loss1 + loss2
# def custom_loss(y2, y1, y_real):
#     loss1 = torch.mean((y2-y1) ** 2)
#     loss2 = torch.mean((y_real - y2) ** 2)
#     return loss1 + loss2

# 计算并打印评价指标
def calculate_metrics(y1_list, y_real_list):
    mse = mean_squared_error(y_real_list, y1_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real_list, y1_list)
    r2 = r2_score(y_real_list, y1_list)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

def save_predictions_to_excel(y1_list, y_real_list, filename="predictions_vs_real.xlsx"):
    # 创建一个 DataFrame
    data = {
        "Predicted y1": y1_list,
        "Real y": y_real_list
    }
    df = pd.DataFrame(data)

    # 将 DataFrame 保存到 Excel 文件
    df.to_excel(filename, index=False)
    print(f"Predictions and real values saved to {filename}")