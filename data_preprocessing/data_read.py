import pandas as pd
import os
import glob

"""该文件的主要目的是读取传感器数据和电性能数据，进行预处理并合并"""

def state_num(df):
    for index , row in df.iterrows():
        if row['类型'] == '电加载中-测试导通电阻（大电流模式）':
            df.at[index,'类型'] = 0
        elif row['类型'] == '电性能加载后-测试关断电阻（降温中）':
            df.at[index,'类型'] = 1
        elif row['类型'] == '电性能加载后-测试导通电阻（降温中）':
            df.at[index,'类型'] = 2
        else :
            df.at[index,'类型'] = 3

    return df

def calculate_cycles(df):
    """
    计算大循环周期和小循环关断导通次数
    """
    df['循环大周期'] = 0
    df['小循环_关断导通次数'] = 0

    cycle_number = 1
    in_cycle = False

    for index, row in df.iterrows():
        # 当前行是 '电加载中-测量加载值'
        if row['类型'] == '电加载中-测量加载值':
            # 检查下一行是否存在，并且包含 '电加载中-测试导通电阻（大电流模式-初值）' 或 '电加载中-测试导通电阻（大电流模式）'
            if index + 1 < len(df) and (
                    '电加载中-测试导通电阻（大电流模式-初值）' in df.at[index + 1, '类型'] or
                    '电加载中-测试导通电阻（大电流模式）' in df.at[index + 1, '类型']
            ):
                if not in_cycle:
                    in_cycle = True
                else:
                    cycle_number += 1
        df.at[index, '循环大周期'] = cycle_number

    for cycle in df['循环大周期'].unique():
        cycle_df = df[df['循环大周期'] == cycle]
        small_cycle_count = 0
        last_event = None

        for index, row in cycle_df.iterrows():
            if row['类型'] == '电性能加载后-测试关断电阻（降温中）':
                if last_event != '关断电阻':
                    small_cycle_count += 1
                    last_event = '关断电阻'
                df.at[index, '小循环_关断导通次数'] = small_cycle_count
            elif row['类型'] == '电性能加载后-测试导通电阻（降温中）':
                if last_event != '导通电阻':
                    last_event = '导通电阻'
                df.at[index, '小循环_关断导通次数'] = small_cycle_count

    return df


def process_and_round_electric_data(df):
    """
    四舍五入采集时刻(s)列并删除重复的采集时刻
    """
    df['采集时刻(s)'] = df['采集时刻(s)'].round().astype(int)
    #df = df.drop_duplicates(subset=['采集时刻(s)'], keep='last')
    return df


def process_electric_data(file_path):
    """
    读取电性能数据，删除了类型为电性能加载前|恢复室温的行并把类型列中的NaN换为空,还取整了采样时刻的值
    """
    df = pd.read_csv(file_path, encoding='gbk')
    print(df.columns)  # 打印列名，确认电阻值(Ω) 列是否存在
    if '电阻值(Ω)' not in df.columns:
        print("电阻值(Ω) 列不存在")

    #df['类型'] = df['类型'].fillna('')  # 将类型列里的 NaN 值替换为空字符串但类型中好像并没有NaN
    # df['电阻值(Ω)'] = df['电阻值(Ω)'].fillna('')  # 将 NaN 值替换为 999999
    df['电阻值(Ω)'] = df['电阻值(Ω)']  # 保留原始 NaN 值（相当于没做处理）

    df = df[~df['类型'].str.contains('电性能加载前|恢复室温')]
    df = calculate_cycles(df)
    df = process_and_round_electric_data(df)
    df = state_num(df)
    return df


def process_sensor_data(sensor_file_path):
    """
    读取并处理传感器数据,处理包括合并了类型和通道这两列(并排列好顺序)和取整采集时刻的数值,返回一个DataFrame二元数组
    """
    sensor_df = pd.read_csv(sensor_file_path, encoding='gbk')
    sensor_df['采集时刻(s)'] = sensor_df['采集时刻(s)'].round().astype(int)
    #sensor_df = sensor_df.drop_duplicates(subset=['采集时刻(s)'], keep='last')
    sensor_df['类型-通道'] = sensor_df['类型'] + sensor_df['通道']
    sensor_df = sensor_df.drop(columns=['类型', '通道'])
    sensor_df = sensor_df[['序号', '采集时刻(s)', '类型-通道', '均值', '波动峰峰值', '波动占比绝对值%']]
    return sensor_df


def merge_sensor_and_electric(sensor_df, electric_df):
    """
    合并传感器数据和电性能数据，找到第一个和最后一个相同采集时刻的区间，
    并在这个区间内按照60秒的时间窗口划分数据。保留原有的两个采集时刻。
    在每个60秒的时间窗口内,如果电性能数据有多个记录，则取最后一个电性能记录。
    如果在t2有相同采集时刻(s)的数据，则取最后一个，期间其他相同采集时刻(s)
    对于传感器数据的处理只有通道排序和60s窗口划分
    而电性能数据则是60s窗口划分并保留同时刻最后一个的值,然后加入了大循环是否类型标记(不清楚是什么作用)
    """
    merged_sensor_data = []
    merged_electric_data = []
    cycle_number = 1

    # 定义通道顺序
    channel_order = [
        '桥式通道1', '桥式通道2', '桥式通道3', '桥式通道4', '桥式通道5',
        '桥式通道6', '桥式通道7', '桥式通道8', '桥式通道9', '桥式通道10',
        '桥式通道11', '桥式通道12', '桥式通道13', '桥式通道14',
        '单电阻通道1', '单电阻通道2', '单电阻通道3', '单电阻通道4',
        '单电阻通道5', '单电阻通道6', '单电阻通道7', '单电阻通道8',
        '单电阻通道9', '单电阻通道10', '单电阻通道11', '单电阻通道12',
        '单电阻通道13', '单电阻通道14', '单电阻通道15'
    ]

    # 找到传感器和电性能数据中最小和最大相同的采集时刻
    min_time = max(sensor_df['采集时刻(s)'].min(), electric_df['采集时刻(s)'].min())
    max_time = min(sensor_df['采集时刻(s)'].max(), electric_df['采集时刻(s)'].max())
    print( 'min_time:',min_time,'max_time:' ,max_time)
    # 如果最小值和最大值之间的差距小于60秒，则停止操作
    if max_time - min_time < 60:
        print("区间小于60秒，停止操作")
        return pd.DataFrame(), pd.DataFrame()

    # 遍历从最小时间到最大时间，进行数据的划分
    t1 = min_time
    while t1 < max_time:
        t2 = t1 + 60
        if t2 > max_time:
            break  # 如果 t2 超过了最大采集时刻，停止循环

        # 在这个时间范围 [t1, t2] 内截取传感器的数据
        sensor_in_range = sensor_df[(sensor_df['采集时刻(s)'] >= t1) & (sensor_df['采集时刻(s)'] <t2)].copy()
        if sensor_in_range.empty:
            break  # 如果没有传感器数据，则停止

        sensor_in_range['60s划分标记'] = cycle_number

        # 按照定义的通道顺序进行排序
        sensor_in_range['类型-通道'] = pd.Categorical(sensor_in_range['类型-通道'], categories=channel_order, ordered=True)
        sensor_in_range = sensor_in_range.sort_values('类型-通道')

        # 在这个时间范围 [t1, t2] 内截取电性能数据
        electric_in_range = electric_df[(electric_df['采集时刻(s)'] >= t1) & (electric_df['采集时刻(s)'] < t2)].copy()
        if electric_in_range.empty:
            break  # 如果没有电性能数据，则停止

        # 检查 t2 是否有相同的采集时刻数据
        if (electric_in_range['采集时刻(s)'] == t2).sum() > 1:
            # 取最后一个 t2 时刻的记录，并删除其他 t2 时刻的记录
            electric_in_range = pd.concat([electric_in_range,
                                           electric_in_range[electric_in_range['采集时刻(s)'] == t2].iloc[-1:]])

            electric_in_range = electric_in_range.drop_duplicates(subset=['采集时刻(s)'], keep='last')

        electric_in_range['60s划分标记'] = cycle_number
        '''
        # 添加类型标记逻辑
        electric_in_range['类型'] = electric_in_range['类型'].fillna('')  # 替换 NaN 值为空字符串
        
        不清楚大循环标记类型的作用，先不做处理
        electric_in_range['大循环是否类型标记'] = 0  # 初始化大循环标记

        # 设置类型标记
        cycle_df = electric_in_range
        if cycle_df['类型'].str.contains('电加载中-测试导通电阻（大电流模式）').any():
            electric_in_range['大循环是否类型标记'] = 0

        if (cycle_df['类型'].str.contains('电性能加载后-测试关断电阻（降温中）').any() or
                cycle_df['类型'].str.contains('电性能加载后-测试导通电阻（降温中）').any()):
            electric_in_range['大循环是否类型标记'] = 1

        if (cycle_df['类型'].str.contains('电加载中-测试导通电阻（大电流模式）').any() and
                (cycle_df['类型'].str.contains('电性能加载后-测试关断电阻（降温中）').any() or
                 cycle_df['类型'].str.contains('电性能加载后-测试导通电阻（降温中）').any())):
            electric_in_range['大循环是否类型标记'] = 2
        '''

        # 只保留指定的列并按顺序排列
        sensor_in_range = sensor_in_range[[
            '序号', '采集时刻(s)', '60s划分标记', '类型-通道', '均值', '波动峰峰值', '波动占比绝对值%'
        ]]

        electric_in_range = electric_in_range[[
            '序号', '采集时刻(s)', '60s划分标记', '循环大周期', '类型' , '小循环_关断导通次数', '电阻值(Ω)'
        ]]

        # 保存传感器和电性能数据
        merged_sensor_data.append(sensor_in_range)
        merged_electric_data.append(electric_in_range)

        # 滑动窗口，增加 t1 和 t2，并更新划分标记
        t1 += 1
        cycle_number += 1

    # 将列表转换为 DataFrame
    merged_sensor_data = pd.concat(merged_sensor_data, ignore_index=True)
    merged_electric_data = pd.concat(merged_electric_data, ignore_index=True)

    return merged_sensor_data, merged_electric_data

def save_to_excel(merged_df, output_file_path):
    """
    保存合并后的数据到Excel文件，将NaN显示为'NaN'字符串
    """
    # 检查 '电阻值(Ω)' 列是否存在
    if '电阻值(Ω)' in merged_df.columns:
        print("电阻值(Ω) 列的 NaN 值数量: ", merged_df['电阻值(Ω)'].isna().sum())

    # 只对非 'category' 类型的列执行 fillna
    #non_categorical_columns = merged_df.select_dtypes(exclude=['category']).columns
    #merged_df[non_categorical_columns] = merged_df[non_categorical_columns].fillna('NaN')

    merged_df.to_excel(output_file_path, index=False)
    print(f"合并后的文件已保存为: {output_file_path}")


def data_merge(sensor_file_path, electric_file_path, output_file_path1,output_file_path2):
    """
    主函数，处理和合并数据，最后保存结果
    """
    # 处理传感器和电性能数据
    sensor_df = process_sensor_data(sensor_file_path)
    electric_df = process_electric_data(electric_file_path)

    # 合并数据
    merged_sensor_data, merged_electric_data = merge_sensor_and_electric(sensor_df, electric_df)
    print("合并后的电性能数据列名：", merged_electric_data.columns)

    # 保存合并后的数据
    save_to_excel(merged_sensor_data, output_file_path1)
    save_to_excel(merged_electric_data, output_file_path2)

def extract_timestamp(file_name):
    # 这里修改提取规则
    start_idx = file_name.find("传感器") + len("传感器")
    if "大电流加载故障注入实验" in file_name:
        end_idx = file_name.find("大电流加载故障注入实验")
    else:
        end_idx = len(file_name)  # 默认到文件末尾

    if start_idx != -1 and end_idx != -1:
        return file_name[start_idx:end_idx].strip('-')  # 去掉多余的"-"并返回时间戳

    return None  # 无法提取时间戳

def process_files_in_directory(sensor_dir, electric_dir, output_dir):
    """
    遍历传感器和电性能文件夹，匹配中间时间戳相同的文件，并调用data_merge函数处理。
    删除不匹配的文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建新的文件夹

    # 获取传感器文件列表（文件名中包含"传感器"）
    sensor_files = [f for f in glob.glob(os.path.join(sensor_dir, "*.csv")) if "传感器" in os.path.basename(f)]

    # 获取电性能文件列表（文件名中包含"电性能"）
    electric_files = [f for f in glob.glob(os.path.join(electric_dir, "*.csv")) if "电性能和电加载" in os.path.basename(f)]

    # 存储匹配的传感器和电性能文件对
    matched_files = []

    # 找到所有的匹配对
    for sensor_file in sensor_files:
        # 检查传感器文件大小是否小于4KB
        if os.path.getsize(sensor_file) <0:
            print(f"文件 {sensor_file} 小于4KB，跳过处理。")
            continue

        sensor_file_name = os.path.basename(sensor_file)
        sensor_timestamp = extract_timestamp(sensor_file_name)  # 提取传感器文件中的时间戳

        if not sensor_timestamp:
            print(f"文件名格式不正确，无法提取时间戳: {sensor_file}")
            continue

        # 匹配电性能文件中的相同时间戳
        matching_electric_files = [f for f in electric_files if sensor_timestamp in os.path.basename(f)]

        # 如果找到匹配的电性能文件，存储匹配对
        if matching_electric_files:
            for electric_file in matching_electric_files:
                matched_files.append((sensor_file, electric_file))
        else:
            print(f"未找到与 {sensor_file} 对应的电性能文件，跳过处理。")

    # 处理匹配的文件
    for sensor_file, electric_file in matched_files:
        # 检查电性能文件大小是否小于4KB
        if os.path.getsize(electric_file ) <0:
            print(f"文件 {electric_file } 小于4KB，跳过处理。")
            continue

        sensor_timestamp = extract_timestamp(os.path.basename(sensor_file))
        electric_file_name = os.path.basename(electric_file)
        output_file_name1 = f"传感器{sensor_timestamp}-大电流加载实验.xlsx"
        output_file_name2 = f"电性能{sensor_timestamp}-大电流加载实验.xlsx"

        output_file_path1 = os.path.join(output_dir, output_file_name1)
        output_file_path2 = os.path.join(output_dir, output_file_name2)

        # 调用 data_merge 处理和保存结果
        data_merge(sensor_file, electric_file, output_file_path1,output_file_path2)
        print(f"处理完成: {output_file_path1}")

def create_processed_directory(original_dir):
    """
    创建一个与原来目录相同名称加上 _processed 后缀的新目录，用于保存处理后的文件。
    """
    parent_dir, folder_name = os.path.split(original_dir)
    # parent_dir=r'D:\desktop\data_processing(3)\data_processing\试验数据集\result1'
    parent_dir= r'D:\cy\集电与太赫兹中心故障预测\code_test_variables\data_preprocessing'
    # 分割父目录和当前文件夹名
    processed_dir = os.path.join(parent_dir, folder_name + "_processed")  # 在父目录下创建 *_processed 文件夹
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)  # 创建新的文件夹
    return processed_dir

def data_make(input_dir):
    """
    遍历该输入目录下的所有文件，并对每个文件进行处理,仅用于检测文件夹中是否有以.csv结尾的文件
    """
    # 查找所有的 .csv 文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    print(f"检测到的 CSV 文件: {csv_files}")

    # 如果没有检测到 CSV 文件，提示并返回
    if not csv_files:
        print("未找到任何 CSV 文件")
        return

    # 为传感器和电性能数据文件分别创建文件夹
    sensor_dir = input_dir
    electric_dir = input_dir

    # 创建用于存储处理后的文件的新目录
    output_dir = create_processed_directory(input_dir)

    # 处理文件
    process_files_in_directory(sensor_dir, electric_dir, output_dir)


def main():
    # 调用示例
    # input_dir = r'D:\desktop\data_processing(3)\data_processing\试验数据集'
    input_dir = r'D:\cy\集电与太赫兹中心故障预测\code_test_variables\data_preprocessing\raw_data'
    print(os.listdir(input_dir))
    data_make(input_dir)

if __name__ == "__main__":
    main()