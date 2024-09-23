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

# 绘制预测值和真实值的折线图
def plot_predictions_vs_real(y1_list, y_real_list):
    plt.figure(figsize=(10, 6))
    plt.plot(y1_list, label='Predicted y1', color='blue', linestyle='--')
    plt.plot(y_real_list, label='Real y', color='green', linestyle='-')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predicted y1 vs Real y')
    plt.ylim(0.005, 0.04)  # 设置纵坐标范围为0.010到0.025
    plt.legend()
    plt.show()

#局部方法图像
def plot_zoomed_predictions_vs_real(y1_list, y_real_list, start, end):

    if start < 0 or end > len(y1_list) or start >= end:
        print("Invalid start or end indices.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(start, end), y1_list[start:end], label='Predicted y1', color='blue', linestyle='--')
    plt.plot(range(start, end), y_real_list[start:end], label='Real y', color='green', linestyle='-')
    plt.xlabel('Sample (Zoomed)')
    plt.ylabel('Value')
    plt.title(f'Zoomed Predicted y1 vs Real y (Samples {start} to {end})')
    plt.legend()
    plt.show()

