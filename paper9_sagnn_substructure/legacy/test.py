# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import yaml
import torch
# from core.train_helper import run
from core.model import SAGNN

from torch_geometric.datasets import ZINC
from core.utils import calculate_stats
from torch.optim import Adam
from torch_geometric.data import DataLoader
import os

# 设置Graphviz的路径
graphviz_path = 'C:\\Program Files\\Graphviz\\bin'
os.environ["PATH"] += os.pathsep + graphviz_path
from torchviz import make_dot
import os.path as osp
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from core.config import cfg, update_cfg
from core.model import SAGNN
from core.transform import SubgraphsTransform
from torch_geometric.datasets import ZINC
from core.utils import calculate_stats
import os
import pickle
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
import networkx as nx

import matplotlib.pyplot as plt

def load_and_process_data(root, subset, split, transform):
    # 假设 pickle 文件的存放路径遵循某种约定
    pickle_path = osp.join(root, f'{split}.pickle')

    # 加载数据
    with open(pickle_path, 'rb') as f:
        molecules = pickle.load(f)

    # 可能还需要加载一个索引文件，如果在使用数据子集
    if subset:
        index_path = osp.join(root, f'{split}.index')
        with open(index_path, 'r') as f:
            indices = [int(x) for x in f.read().strip().split(',')]
        # 使用索引过滤分子
        molecules = [molecules[i] for i in indices]

    # 处理数据
    data_list = []
    for mol in molecules:
        # 创建 Data 对象
        node_features = torch.tensor(mol['atom_type'], dtype=torch.long).view(-1, 1)
        adj = mol['bond_type']
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

        x = node_features
        y = mol['logP_SA_cycle_normalized'].to(torch.float)

        # 注意这里，我们设置 num_nodes
        num_nodes = x.size(0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
        if transform:
            data = transform(data)
        data_list.append(data)

    return data_list

def create_dataset(cfg):
    root = 'data/triple'  # 根目录路径
    processed_dir = './data/ZINC/subset/processed/processed'
    transform = SubgraphsTransform(cfg)  # 数据转换

    train_path = osp.join(processed_dir, 'train_processed.pt')
    val_path = osp.join(processed_dir, 'val_processed.pt')
    test_path = osp.join(processed_dir, 'test_processed.pt')

    # 检查是否已经处理过数据
    if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path):
        print("Loading preprocessed data...")
        train_dataset = torch.load(train_path)
        val_dataset = torch.load(val_path)
        test_dataset = torch.load(test_path)
    else:
        print("Processing data...")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        # 处理训练、验证和测试数据集
        train_dataset = load_and_process_data(root, subset=False, split='train', transform=transform)
        val_dataset = load_and_process_data(root, subset=False, split='val', transform=transform)
        test_dataset = load_and_process_data(root, subset=False, split='test', transform=transform)

        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)

    return train_dataset, val_dataset, test_dataset


def test(loader, model,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf


def create_model(cfg):
    model = SAGNN(None, None,
                        nhid=cfg.model.hidden_size,#隐藏层大小
                        nout=1,#模型输出维度
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=[cfg.model.gnn_type],
                        embedding_types=cfg.model.embedding_type,#节点或图的嵌入版本
                        hop_dim=cfg.model.hops_dim,
                        fullgraph_pos_enc_dim=cfg.model.fullgraph_pos_enc_dim,
                        egograph_pos_enc_dim=cfg.model.egograph_pos_enc_dim,
                        cutgraph_pos_enc_dim=cfg.model.cutgraph_pos_enc_dim,
                        pos_enc_dim=cfg.model.pos_enc_dim,
                        embedding_learnable=cfg.model.embedding_learnable,
                        pooling=cfg.model.pool,
                        global_embedding=cfg.model.global_embedding,
                        dropout=cfg.train.dropout
)
    # print(model)
    return model

from torch.nn.functional import l1_loss


def train(train_loader, model, optimizer,  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    total_loss = 0
    total_mae = 0  # 用于累积MAE
    N = 0
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs

        optimizer.zero_grad()
        output = model(data)
        loss = (output.squeeze() - y).abs().mean()  # 确保loss计算时形状一致
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_graphs

        # 计算MAE，确保output和y形状一致
        mae = l1_loss(output.squeeze(), y, reduction='mean')  # 使用.squeeze()来匹配y的形状
        total_mae += mae.item() * num_graphs

        N += num_graphs

    avg_loss = total_loss / N
    avg_mae = total_mae / N
    return avg_loss, avg_mae

def test_create_dataset():
    # 测试函数
    global cfg  # 指明我们要使用的是全局变量cfg
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
    print("训练数据集的大小:", len(train_dataset))
    print("验证数据集的大小:", len(val_dataset))
    print("测试数据集的大小:", len(test_dataset))

    # 打印训练数据集中第一个图的一些信息
    if len(train_dataset) > 0:
        data = train_dataset[0]  # 取训练数据集的第一个图

        # 打印图的一些基本信息
        print(f"\n第一个图的信息:")
        print(f"节点数量: {data.num_nodes}")
        print(f"边的数量: {data.num_edges}")
        print(f"平均节点度: {data.num_edges / data.num_nodes:.2f}")
        print(f"是否包含孤立节点: {data.contains_isolated_nodes()}")
        print(f"是否包含自环: {data.contains_self_loops()}")
        print(f"是否是无向图: {data.is_undirected()}")

        # 打印节点特征和边的连接信息的部分内容，这里只展示前5个
        print("节点特征 (x) - 前5个节点:", data.x[:5])
        print("边的连接信息 (edge_index) - 前5条边:\n", data.edge_index[:, :5])

        # 如果数据包含标签，则打印第一个图的标签
        if 'y' in data:
            print("标签 (y):", data.y)



import matplotlib.pyplot as plt

def test_create_dataset_and_model():
    # 更新配置
    global cfg
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)

    # 打印配置信息
    print("Configuration:", cfg)

    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
    print(f"训练数据集大小: {len(train_dataset)}, 验证数据集大小: {len(val_dataset)}, 测试数据集大小: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 创建模型
    model = create_model(cfg).to(cfg.device)
    model.reset_parameters()
    print("模型初始化完成。")

    # 设置优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

    # 初始化记录训练损失和性能指标的列表
    train_losses, val_performances, test_performances = [], [], []

    # 训练和评估模型
    for epoch in range(1, cfg.train.epochs + 1):
        # 训练
        model.train()
        train_loss, _ = train(train_loader, model, optimizer)
        scheduler.step()

        # 验证
        model.eval()
        val_perf = test(val_loader, model)

        # 测试
        test_perf = test(test_loader, model)

        # 记录结果
        train_losses.append(train_loss)
        val_performances.append(val_perf)
        test_performances.append(test_perf)

        # 打印和记录结果
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val: {val_perf:.4f}, Test: {test_perf:.4f}')

    # 可选: 保存模型
    torch.save(model, f'model_final.pt')
    print("模型训练完成并保存至 model_final.pt")

    # 可视化训练损失和性能指标
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_performances, label='Validation Performance')
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_performances, label='Test Performance')
    plt.title('Test Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()

    plt.tight_layout()
    plt.show()


def exp_pickle():
    import pickle

    # 只需一个文件路径
    file_path = r'C:\Users\BDY\Desktop\Triple\data\struc_result.pkl'
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\raw\train.pickle'

    # 加载.pkl文件
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 检查data的类型
    print(f"Loaded data type: {type(data)}")

    # 如果data是字典类型
    if isinstance(data, dict):
        # 打印所有键和值（考虑到数据量可能很大，这里限制了打印数量）
        for i, (key, value) in enumerate(data.items()):
            print(f"Key: {key}")
            if isinstance(value, (list, tuple, dict)):
                # 如果值是一个容器类型，打印容器的长度或部分内容
                print(f"  Value type: {type(value)}, Length: {len(value)}")
                if isinstance(value, dict):
                    for sub_key, sub_value in list(value.items())[:5]:  # 打印前5个子元素
                        print(f"    Subkey: {sub_key}, Subvalue: {sub_value}")
                elif isinstance(value, (list, tuple)):
                    print("  Values:", value[:5])  # 打印前5个元素
            else:
                # 直接打印值
                print(f"  Value: {value}")

    # 如果data是列表或元组
    elif isinstance(data, (list, tuple)):
        # 遍历前5个元素（如果列表非常大，这里假设前5个足够代表）
        for i, element in enumerate(data[:5], start=1):
            print(f"Element {i}:")
            if isinstance(element, dict):
                for key, value in element.items():
                    # 对于字典的每个键值对，尝试打印详细信息
                    print(f"  Key: {key}, Value: {value}")
            else:
                # 如果元素不是字典，直接打印
                print(f"  Value: {element}")

    else:
        # 如果data是其他类型
        print("Data is neither a list nor a dictionary.")
        print("Data:", data)

import torch

import torch


# Execute the function

# Call the function to view the contents of the .pt file




if __name__ == "__main__":
    # convert_pt_to_pkl( )
    #
    # review_model()
    test_create_dataset_and_model()

    # view_pt()
    # exp_pickle()
    # test_create_dataset_EXP()
    test_create_dataset()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader, Data

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, num_classes)  # 确保 num_classes 正确

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


def load_data(file_path):
    # 加载.pt文件
    data_list = torch.load(file_path)
    data_objects = []
    for data in data_list:
        data_obj = Data(
            x=data['x'].to(torch.float),  # 确保节点特征是浮点型
            edge_index=data['edge_index'].reshape(2, -1),  # 转换成 [2, 边的数量]
            edge_attr=data['edge_attr'],
            y=data['y'].to(torch.long)  # 确保标签是长整型
        )
        data_objects.append(data_obj)
    return data_objects


def main():
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\triple\subset\processed\processed\train_processed_ord.pt'
    dataset = load_data(file_path)

    # 取得所有不同标签的数量，用于设置模型输出层
    all_labels = torch.cat([data.y for data in dataset])
    num_classes = all_labels.max().item() + 1  # 标签从0开始，所以取最大值+1

    model = GCN(num_node_features=dataset[0].num_features,
                num_classes=num_classes,
                hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(200):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            if batch.y.max() >= num_classes:
                raise ValueError("One of the batch labels is out of model's output range.")
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        _, pred = model(batch.x, batch.edge_index, batch.batch).max(dim=1)
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()

