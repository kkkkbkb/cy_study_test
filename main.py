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

from models.FC_LSTM import SensorNet, ElectricNet
from utils.plot_figs import plot_predictions_vs_real, plot_zoomed_predictions_vs_real
from utils.utils_model import custom_loss, calculate_metrics, save_predictions_to_excel, save_predictions_to_excel

from config import get_args

args = get_args()

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""DATA"""
def generate_data(sensor_file, electric_file, y_real_file, train_ratio=0.8):
    # 读取传感器数据
    sensor_df = pd.read_excel(sensor_file)
    # 读取电性能数据
    electric_df = pd.read_excel(electric_file)
    # 读取真实值数据
    y_real_df = pd.read_excel(y_real_file)

    # 标准化传感器数据
    scaler = StandardScaler()
    sensor_df[['均值', '波动峰峰值', '波动占比绝对值%']] = scaler.fit_transform(sensor_df[['均值', '波动峰峰值', '波动占比绝对值%']])

    # 提取60s划分标记，用于数据分组
    sensor_groups = sensor_df.groupby('60s划分标记')
    electric_groups = electric_df.groupby('60s划分标记')
    y_real_groups = y_real_df.groupby('60s划分标记')

    # 获取所有的60s划分标记
    all_groups = list(sensor_groups.groups.keys())

    # 按照给定比例划分训练集和测试集
    train_groups, test_groups = train_test_split(all_groups, train_size=train_ratio, random_state=42)

    # 准备存放训练和测试数据的列表
    train_sensor_data_list = []
    train_electric_data_list = []
    train_y_real_list = []

    test_sensor_data_list = []
    test_electric_data_list = []
    test_y_real_list = []

    # 遍历每个 60s 划分的标记组，生成数据
    for i, group in sensor_groups:
        # 提取传感器数据的三列 (均值, 波动峰峰值, 波动占比绝对值%)
        sensor_values = group[['均值', '波动峰峰值', '波动占比绝对值%']].values.flatten()

        # 提取电性能数据
        if i in electric_groups.groups:
            electric_values = electric_groups.get_group(i)['电子负载输入电压(V)'].values
        else:
            electric_values = np.zeros((1,))  # 如果没有该组，填充0

        # 确保电性能数据是数值类型
        electric_values = electric_values.astype(np.float32)

        # 提取真实电阻值
        if i in y_real_groups.groups:
            y_real_value = y_real_groups.get_group(i)['电子负载输入电压(V)'].values[0]
        else:
            y_real_value = 0.0  # 如果没有真实电阻值，填充0

        # # 对于训练集：检查真实值和电性能数据的电阻是否为 NaN，剔除包含 NaN 的样本
        # if i in train_groups:
        #     if not np.isnan(y_real_value) and not np.isnan(electric_values).any():  # 检查电阻值是否为 NaN
        #         train_sensor_data_list.append(sensor_values)
        #         train_electric_data_list.append(torch.tensor(electric_values, dtype=torch.float32))
        #         train_y_real_list.append(torch.tensor([y_real_value], dtype=torch.float32))

        # 对于训练集：将 NaN 替换为 0.1，保持与测试集一致的处理
        if i in train_groups:
            sensor_values_cleaned = np.where(np.isnan(sensor_values), 0.1, sensor_values)
            electric_values_cleaned = np.where(np.isnan(electric_values), 0.1, electric_values)
            y_real_value_cleaned = 0.1 if np.isnan(y_real_value) else y_real_value

            train_sensor_data_list.append(sensor_values_cleaned)
            train_electric_data_list.append(torch.tensor(electric_values_cleaned, dtype=torch.float32))
            train_y_real_list.append(torch.tensor([y_real_value_cleaned], dtype=torch.float32))

        # 对于测试集：将电阻值为 NaN 的样本替换为 0.1
        elif i in test_groups:
            test_sensor_data_list.append(sensor_values)

            # 清理电性能数据中的 NaN 值，替换为 0.1
            electric_values_cleaned = np.where(np.isnan(electric_values), 0.1, electric_values)

            # 检查真实电阻值是否为 NaN，若是则替换为 0.1
            y_real_value_cleaned = 0.1 if np.isnan(y_real_value) else y_real_value

            test_electric_data_list.append(torch.tensor(electric_values_cleaned, dtype=torch.float32))
            test_y_real_list.append(torch.tensor([y_real_value_cleaned], dtype=torch.float32))

    # 将传感器数据转换为 PyTorch 张量
    train_sensor_data = torch.tensor(np.array(train_sensor_data_list), dtype=torch.float32)
    test_sensor_data = torch.tensor(np.array(test_sensor_data_list), dtype=torch.float32)

    return (train_sensor_data, train_electric_data_list, train_y_real_list), \
           (test_sensor_data, test_electric_data_list, test_y_real_list)

# def custom_collate_fn(batch):
#     sensor_data, electric_data, y_real = zip(*batch)
#     sensor_data = torch.stack(sensor_data)
#     electric_data = [torch.tensor(e, dtype=torch.float32).unsqueeze(-1) if e.dim() == 1 else e for e in electric_data]
#     electric_data_padded = pad_sequence(electric_data, batch_first=True)
#     y_real = torch.stack(y_real)
#     return sensor_data, electric_data_padded, y_real
def custom_collate_fn(batch):
    sensor_data, electric_data, y_real = zip(*batch)

    # 将 numpy.ndarray 转换为 torch.Tensor
    sensor_data = [torch.tensor(sd, dtype=torch.float32) for sd in sensor_data]

    # 将 sensor_data 转换为 torch.Tensor，并进行堆叠
    sensor_data = torch.stack(sensor_data)

    # 对 electric_data 进行适当处理
    electric_data = [torch.tensor(e, dtype=torch.float32).unsqueeze(-1) if isinstance(e, np.ndarray) and e.ndim == 1 else e for e in electric_data]
    electric_data_padded = pad_sequence(electric_data, batch_first=True)

    # 将 y_real 也转换为 Tensor
    y_real = torch.stack(y_real)

    return sensor_data, electric_data_padded, y_real

class CustomDataset(Dataset):
    def __init__(self, sensor_data, electric_data_list, y_real_list):
        self.sensor_data = sensor_data
        self.electric_data_list = electric_data_list
        self.y_real_list = y_real_list

    def __len__(self):
        return len(self.sensor_data)

    # def __getitem__(self, idx):
    #     return self.sensor_data[idx], self.electric_data_list[idx], self.y_real_list[idx]
    def __getitem__(self, idx):
        sensor_data_cleaned = np.where(np.isnan(self.sensor_data[idx]), 0.1, self.sensor_data[idx])
        electric_data_cleaned = np.where(np.isnan(self.electric_data_list[idx]), 0.1, self.electric_data_list[idx])
        return sensor_data_cleaned, electric_data_cleaned, self.y_real_list[idx]

"""TRAIN"""
def train(sensor_net, electric_net, dataloader, optimizer, epochs=100):
    sensor_net.train()
    electric_net.train()

    for epoch in range(epochs):
        total_loss = 0
        for sensor_data, electric_data, y_real in dataloader:
            sensor_data, electric_data, y_real = sensor_data.to(device), electric_data.to(device), y_real.to(device)  # 移动数据到GPU

            optimizer.zero_grad()

            sensor_feat, y1 = sensor_net(sensor_data)
            y2 = electric_net(sensor_feat, electric_data)

            loss = custom_loss(y2, y1, y_real)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

"""TEST"""
def test(sensor_net, electric_net, dataloader):
    sensor_net.eval()
    electric_net.eval()

    y1_list = []
    y_real_list = []

    with torch.no_grad():
        for sensor_data, electric_data, y_real in dataloader:
            sensor_data, electric_data, y_real = sensor_data.to(device), electric_data.to(device), y_real.to(device)  # 移动数据到GPU

            sensor_feat, y1 = sensor_net(sensor_data)

            y1_list.extend(y1.squeeze().tolist())
            y_real_list.extend(y_real.squeeze().tolist())

    plot_predictions_vs_real(y1_list, y_real_list)
    plot_zoomed_predictions_vs_real(y1_list, y_real_list, 200, 400)  # 示例：放大 200 到 400 号样本
    calculate_metrics(y1_list, y_real_list)
    save_predictions_to_excel(y1_list, y_real_list)

# 主函数
if __name__ == "__main__":
    # sensor_file = r'D:\cy\集电与太赫兹中心故障预测\code_test_vol\data_preprocessing\raw_data_processed\传感器20240812_153727-大电流加载实验.xlsx'
    # electric_file = r'D:\cy\集电与太赫兹中心故障预测\code_test_vol\data_preprocessing\raw_data_processed\电性能20240812_153727-大电流加载实验.xlsx'
    #y_real_file = r'D:\cy\集电与太赫兹中心故障预测\code_test_vol\data_preprocessing\raw_data_processed\y_result_电性能20240812_153727-大电流加载实验.xlsx'

    # 生成训练集和测试集
    (train_sensor_data, train_electric_data_list, train_y_real_list), \
    (test_sensor_data, test_electric_data_list, test_y_real_list) = generate_data(sensor_file = args.sensor_file,
                                                                                  electric_file = args.electric_file,
                                                                                  y_real_file = args.y_real_file,
                                                                                  train_ratio=0.8)

    # 创建数据集
    train_dataset = CustomDataset(train_sensor_data, train_electric_data_list, train_y_real_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    test_dataset = CustomDataset(test_sensor_data, test_electric_data_list, test_y_real_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 初始化网络并移动到GPU
    sensor_net = SensorNet().to(device)
    electric_net = ElectricNet().to(device)

    # 优化器
    optimizer = optim.Adam(list(sensor_net.parameters()) + list(electric_net.parameters()), lr=0.001, weight_decay=1e-5)

    # 训练模型
    print("Training the model...")
    train(sensor_net, electric_net, train_dataloader, optimizer, epochs=100)

    # 测试模型
    print("Testing the model...")
    test(sensor_net, electric_net, test_dataloader)

