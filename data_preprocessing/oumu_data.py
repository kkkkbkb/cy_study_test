import os
import pandas as pd

"""该文件的主要功能是从包含电阻值的 Excel 文件中提取每个分组的电阻值，并将这些数据保存到新的 Excel 文件中"""

def extract_first_resistance_value(file_path):
    """
    从 Excel 文件中提取电阻值,并将每个组的下一个60s时段的第一个值返回
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 去掉列名中的空格
    df.columns = df.columns.str.strip()

    # 确认 '60s划分标记' 和 '电阻值(Ω)' 列存在
    if '60s划分标记' not in df.columns or '电阻值(Ω)' not in df.columns:
        raise ValueError("必要的列 '60s划分标记' 或 '电阻值(Ω)' 不存在，请检查列名。")

    # 按照 '60s划分标记' 分组并取每个组的第一个电阻值
    first_rows = df.groupby('60s划分标记').agg({'电阻值(Ω)': 'first'}).reset_index()

    # 将电阻值向下移动一行，这样每个分组就会取到下一个时段的电阻值
    first_rows['电阻值(Ω)'] = first_rows['电阻值(Ω)'].shift(-1)

    # # 删除空值（可能最后一组会没有下一个时段）
    # first_rows.dropna(subset=['电阻值(Ω)'], inplace=True)
    # first_rows.dropna(subset=['电阻值(Ω)'], inplace=True)

    # 返回结果
    return first_rows

def save_to_excel(merged_df, output_file_path):
    """
    保存合并后的数据到Excel文件，确保NaN显示为'NaN'字符串
    """
    # 检查 NaN 值
    print("处理前的 NaN 数量: ", merged_df.isna().sum())

    # 将 NaN 值替换为 'NaN' 字符串
    merged_df = merged_df.fillna('NaN')

    # 检查是否所有 NaN 已被替换
    print("处理后的 NaN 数量: ", merged_df.isna().sum())

    # 保存到 Excel 文件
    merged_df.to_excel(output_file_path, index=False)
    print(f"合并后的文件已保存为: {output_file_path}")

def process_electric_files(input_dir):
    """
    遍历目录下所有包含“电性能”的xlsx文件，提取每个文件的电阻值并保存结果。
    """
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 检查文件名中是否包含“电性能”，且扩展名为 .xlsx
            if "电性能" in file and file.endswith(".xlsx"):
                file_path = os.path.join(root, file)
                try:
                    # 提取电阻值
                    resistance_data = extract_first_resistance_value(file_path)

                    # 生成新的文件名
                    output_file_name = f"y_result_{file}"
                    output_path = os.path.join(root, output_file_name)

                    # 保存结果到新的Excel文件
                    save_to_excel(resistance_data, output_path)
                    print(f"处理完成并保存为: {output_file_name}")

                except ValueError as e:
                    print(f"文件处理错误: {file} - {e}")

# 调用示例
# input_dir = r'D:\desktop\data_processing(3)\data_processing\试验数据集\result1' # E:\博士阶段文件\5. 项目文件\17. 集电与太赫兹中心故障预测\data_processing(1)\data_processing(3)\data_processing\试验数据集\result1
input_dir = r'D:\cy\集电与太赫兹中心故障预测\code_test\data_preprocessing\raw_data_processed'
process_electric_files(input_dir)