import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Train Net')

    """以下参数要根据情况改变"""
    # 索引
    parser.add_argument('--batch_size', type=int, default=128, help='The number of batch size')

    parser.set_defaults(argument=True)

    return parser.parse_args()