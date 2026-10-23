import pickle
import random

def split_data(pickle_file_path, train_path, val_path, test_path, train_ratio=0.8, val_ratio=0.1):
    # 读取Pickle文件
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)

    # 随机打乱数据
    random.shuffle(data)

    # 计算划分点
    total_count = len(data)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    # 划分数据集
    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]

    # 保存各个数据集为Pickle文件
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

# 指定文件路径
pickle_file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\CEXP\data.pkl'
pickle_file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\MUTAG\mutag_dataset.pkl'
pickle_file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PROTEINS\proteins_dataset.pkl'

train_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PROTEINS\train.pickle'
val_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PROTEINS\val.pickle'
test_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PROTEINS\test.pickle'


# 调用函数进行数据划分
split_data(pickle_file_path, train_path, val_path, test_path)

# import torch
# import pickle
# from torch_geometric.data import Data
#
#
# def load_pickle_and_convert_to_torch(path, save_path):
#     # 读取Pickle文件
#     with open(path, 'rb') as file:
#         data_list = pickle.load(file)
#
#     # 转换数据并保存为.pt文件
#     torch_data_list = []
#     for item in data_list:
#         # 将列表内部的列表转换为tensor，适用于'edge_index'
#         edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
#
#         # 将'x'中的特征转换为tensor
#         x = torch.tensor(item['x'], dtype=torch.float)
#
#         # 将'y'标签转换为tensor
#         y = torch.tensor(item['y'], dtype=torch.float)
#
#         # 创建PyTorch Geometric Data对象
#         torch_data = Data(x=x, edge_index=edge_index, y=y)
#         torch_data_list.append(torch_data)
#
#     # 保存为PT文件
#     torch.save(torch_data_list, save_path)
#
#
# # 定义文件路径
# train_pickle = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\train.pickle'
# val_pickle = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\val.pickle'
# test_pickle = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\test.pickle'
#
# # 定义PT文件保存路径
# train_pt_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\train.pt'
# val_pt_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\val.pt'
# test_pt_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\test.pt'
#
# # 转换并保存文件
# load_pickle_and_convert_to_torch(train_pickle, train_pt_path)
# load_pickle_and_convert_to_torch(val_pickle, val_pt_path)
# load_pickle_and_convert_to_torch(test_pickle, test_pt_path)
