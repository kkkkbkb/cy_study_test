import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Train Net')

    """以下参数要根据情况改变"""
    # 索引
    parser.add_argument('--batch_size', type=int, default=128, help='The number of batch size')
    parser.add_argument('--sensor_file', type=str, 
                        default= r'D:\cy\集电与太赫兹中心故障预测\code_test_variables\data_preprocessing\raw_data_processed\传感器20240812_153727-大电流加载实验.xlsx'
                        ,help='The path of sensor file')
    parser.add_argument('--electric_file', type=str, 
                        default= r'D:\cy\集电与太赫兹中心故障预测\code_test_variables\data_preprocessing\raw_data_processed\电性能20240812_153727-大电流加载实验.xlsx'
                        ,help='The path of electric file')
    parser.add_argument('--y_real_file', type=str, 
                        default= r'D:\cy\集电与太赫兹中心故障预测\code_test_variables\data_preprocessing\raw_data_processed\y_result_电性能20240812_153727-大电流加载实验.xlsx'
                        ,help='The path of y_real file')
    

    parser.set_defaults(argument=True)

    return parser.parse_args()