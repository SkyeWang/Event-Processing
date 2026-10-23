import os

# 设置Graphviz的路径
graphviz_path = 'C:\\Program Files\\Graphviz\\bin'
os.environ["PATH"] += os.pathsep + graphviz_path
import csv
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

def load_and_process_data(root, subset, split, transform=None):
    # 假设 pickle 文件的存放路径遵循某种约定
    pickle_path = osp.join(root, f'{split}.pickle')

    # 加载数据
    with open(pickle_path, 'rb') as f:
        data_list = pickle.load(f)

    # 如果在使用数据子集
    if subset:
        index_path = osp.join(root, f'{split}.index')
        with open(index_path, 'r') as f:
            indices = [int(x) for x in f.read().strip().split(',')]
        # 使用索引过滤数据
        data_list = [data_list[i] for i in indices]

    # 转换数据
    torch_data_list = []
    for item in data_list:
        # 将列表内部的列表转换为tensor，适用于'edge_index'
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long)

        # 将'x'中的特征转换为tensor
        x = torch.tensor(item['x'], dtype=torch.float)

        # 将'y'标签转换为tensor
        y = torch.tensor(item['y'], dtype=torch.float)

        # 计算节点数
        num_nodes = x.size(0)

        # 创建PyTorch Geometric Data对象
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        # 应用可能的转换
        if transform:
            data = transform(data)

        torch_data_list.append(data)

    return torch_data_list

def create_dataset(cfg):
    root = 'data/PROTEINS'  # 根目录路径
    processed_dir = './data/PROTEINS/subset/processed/processed'
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

def train(train_loader, model, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    total_loss = 0
    total_correct = 0  # 用于累积正确分类的数量
    N = 0
    loss_func = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数

    for data in train_loader:
        data, y, num_graphs = data.to(device), data.y.to(device), data.num_graphs
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_graphs
        predicted = (output.sigmoid() > 0.5).float()  # 计算预测结果
        correct_this_batch = (predicted == y.unsqueeze(1)).sum().item()  # 计算本批次的正确预测数量
        total_correct += correct_this_batch  # 累加正确预测数量
        N += num_graphs

    avg_loss = total_loss / N
    accuracy = total_correct / N
    return avg_loss, accuracy

def create_model(cfg):
    model = SAGNN(None, None,
                  nhid=cfg.model.hidden_size,  # 隐藏层大小
                  nout=1,  # 模型输出维度为1，用于二分类
                  nlayer_outer=cfg.model.num_layers,
                  nlayer_inner=cfg.model.mini_layers,
                  gnn_types=[cfg.model.gnn_type],
                  embedding_types=cfg.model.embedding_type,
                  # embedding_types='NONE',
                  hop_dim=cfg.model.hops_dim,
                  fullgraph_pos_enc_dim=cfg.model.fullgraph_pos_enc_dim,
                  egograph_pos_enc_dim=cfg.model.egograph_pos_enc_dim,
                  cutgraph_pos_enc_dim=cfg.model.cutgraph_pos_enc_dim,
                  pos_enc_dim=cfg.model.pos_enc_dim,
                  embedding_learnable=cfg.model.embedding_learnable,
                  pooling=cfg.model.pool,
                  global_embedding=cfg.model.global_embedding,
                  dropout=cfg.train.dropout)
    return model



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
    train_losses, train_accuracies, val_performances, test_performances = [], [], [], []

    # 训练和评估模型
    result_dir = './results_1000'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 打开CSV文件进行写入
    with open(os.path.join(result_dir, 'PROTEINS_SAGNN_results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy'])

        for epoch in range(1, cfg.train.epochs + 1):
            # 训练
            model.train()
            train_loss, train_acc = train(train_loader, model, optimizer, cfg.device)
            scheduler.step()

            # 验证
            model.eval()
            # 测试

            # 记录结果
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            results = []
            results.append([epoch, train_loss, train_acc])

            writer.writerow([epoch, train_loss, train_acc])
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            # 打印和记录结果
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

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
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_performances, label='Validation Performance')
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_create_dataset_and_model_val():
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
    train_losses, train_accuracies, val_performances, test_performances = [], [], [], []

    # 训练和评估模型

    # 打开CSV文件进行写入
    for epoch in range(1, cfg.train.epochs + 1):
        # 训练
        model.train()
        train_loss, train_acc = train(train_loader, model, optimizer, cfg.device)
        scheduler.step()

        # 验证
        model.eval()
        # 测试

        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        results = []
        results.append([epoch, train_loss, train_acc])

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')


if __name__ == "__main__":
    # test_create_dataset_and_model()
    # test_create_dataset()
    test_create_dataset_and_model_val()