import os
import pickle
from torch_geometric.datasets import TUDataset

def save_proteins_as_pkl():
    # 设置数据集的存储路径
    dataset_path = './data/PROTEINS'
    os.makedirs(dataset_path, exist_ok=True)  # 确保目录存在

    # 加载PROTEINS数据集
    dataset = TUDataset(root=dataset_path, name='PROTEINS')

    # 将数据集保存为Pickle文件
    with open(os.path.join(dataset_path, 'proteins_dataset.pkl'), 'wb') as f:
        pickle.dump([data for data in dataset], f)

    print("PROTEINS dataset saved as a pickle file at './data/PROTEINS/proteins_dataset.pkl'.")

if __name__ == '__main__':
    save_proteins_as_pkl()
