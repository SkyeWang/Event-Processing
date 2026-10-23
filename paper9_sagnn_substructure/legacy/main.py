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
import csv
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
def create_dataset(cfg):
    # No need to do offline transforSAGNNmation
    cfg.processed_dir = './data/ZINC/subset/processed/processed'
    transform = SubgraphsTransform(cfg)

    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform)
    test_dataset = ZINC(root, subset=True, split='test', transform=transform)

    # When without randomness, transform the data to save a bit time
    if os.path.exists(cfg.processed_dir):
        train_dataset = torch.load(osp.join(cfg.processed_dir, 'train_processed.pt'))
        val_dataset = torch.load(osp.join(cfg.processed_dir, 'val_processed.pt'))
        test_dataset = torch.load(osp.join(cfg.processed_dir, 'test_processed.pt'))
    else:
        os.makedirs(cfg.processed_dir)
        train_dataset = [x for x in train_dataset]
        val_dataset = [x for x in val_dataset]
        test_dataset = [x for x in test_dataset]
        torch.save(train_dataset, osp.join(cfg.processed_dir, 'train_processed.pt'))
        torch.save(val_dataset, osp.join(cfg.processed_dir, 'val_processed.pt'))
        torch.save(test_dataset, osp.join(cfg.processed_dir, 'test_processed.pt'))

    print('------------Train--------------')
    calculate_stats(train_dataset)
    print('------------Validation--------------')
    calculate_stats(val_dataset)
    print('------------Test--------------')
    calculate_stats(test_dataset)
    print('------------------------------')

    return train_dataset, val_dataset, test_dataset


def test(loader, model,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
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

def create_dataset_EXP(cfg):
    # No need to do offline transforSAGNNmation
    cfg.processed_dir = './data/EXP/subset/processed/processed'
    transform = SubgraphsTransform(cfg)
    root = 'data/EXP'
    train_dataset = ZINC(root=root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root=root, subset=True, split='val', transform=transform)
    test_dataset = ZINC(root=root, subset=True, split='test', transform=transform)

    # When without randomness, transform the data to save a bit time
    if os.path.exists(cfg.processed_dir):
        train_dataset = torch.load(osp.join(cfg.processed_dir, 'train_processed.pt'))
        val_dataset = torch.load(osp.join(cfg.processed_dir, 'val_processed.pt'))
        test_dataset = torch.load(osp.join(cfg.processed_dir, 'test_processed.pt'))
    else:
        os.makedirs(cfg.processed_dir)
        train_dataset = [x for x in train_dataset]
        val_dataset = [x for x in val_dataset]
        test_dataset = [x for x in test_dataset]
        torch.save(train_dataset, osp.join(cfg.processed_dir, 'train_processed.pt'))
        torch.save(val_dataset, osp.join(cfg.processed_dir, 'val_processed.pt'))
        torch.save(test_dataset, osp.join(cfg.processed_dir, 'test_processed.pt'))

    print('------------Train--------------')
    print(f'Train Dataset: {len(train_dataset)} samples')
    print('------------Validation--------------')
    print(f'Validation Dataset: {len(val_dataset)} samples')
    print('------------Test--------------')
    print(f'Test Dataset: {len(test_dataset)} samples')
    print('------------------------------')

    return train_dataset, val_dataset, test_dataset

def test_create_dataset_EXP():
    # 测试函数
    global cfg  # 指明我们要使用的是全局变量cfg
    # 这里假设您已经有一个针对EXP数据集的配置文件，或者您可以在这里直接修改cfg以适应EXP数据集
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)

    # 使用EXP专用的创建数据集函数
    train_dataset, val_dataset, test_dataset = create_dataset_EXP(cfg)
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

def create_model(cfg):
    model = SAGNN(None, None,
                  nhid=cfg.model.hidden_size,  #隐藏层大小
                  nout=1,  #模型输出维度
                  nlayer_outer=cfg.model.num_layers,
                  nlayer_inner=cfg.model.mini_layers,
                  gnn_types=[cfg.model.gnn_type],
                  embedding_types=cfg.model.embedding_type,  #节点或图的嵌入版本
                  # embedding_types='None',  # 节点或图的嵌入版本
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

    # 收集所有标签来确定数据集是否适合分类任务
    all_labels = []
    for data in train_dataset:
        all_labels.append(data.y.item())  # 假设y是标量值，用于分类标签

    unique_labels = set(all_labels)
    print(f"唯一标签（类别）数: {len(unique_labels)}")
    print(f"唯一标签（类别）: {unique_labels}")
    print(unique_labels)
    if len(unique_labels) > 2:
        print("这是一个多分类任务")
    elif len(unique_labels) == 2:
        print("这是一个二分类任务")
    else:
        print("数据集不适用于分类任务")
    # 打印训练数据集中第一个图的一些信息（可选）
    if len(train_dataset) > 0:
        data = train_dataset[0]
        # 这里可以添加打印信息的代码
        print(f"\n第一个图的信息:")
        print(f"节点数量: {data.num_nodes}")
        print(f"边的数量: {data.num_edges}")
        # 这里继续添加其他打印信息

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
    with open(os.path.join(result_dir, 'ZINC_SAGNN_results.csv'), 'w', newline='') as file:
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

    # 可选: 保存模型
    torch.save(model, f'ZINC_SAGNN_500.pt')
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


def exp_pickle():
    import pickle

    # 只需一个文件路径
    file_path = r'C:\Users\BDY\Desktop\Triple\data\struc_result.pkl'
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\raw\train.pickle'
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\test.pickle'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\triple\train.pickle'
    # file_path = r'C:\Users\BDY\Desktop\Triple\data\splits\test.pickle'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\MUTAG\mutag_dataset.pkl'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PTC\ptc_dataset.pkl'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PROTEINS\proteins_dataset.pkl'

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

def view_pt():
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\subset\processed\processed\train_processed.pt'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\triple\processed\train.pt'
    data = torch.load(file_path)

    def print_tensor_details(tensor, name, indent):
        prefix = ' ' * indent
        print(f"{prefix}{name}: shape={tensor.shape}, dtype={tensor.dtype}")
        # Adjust the threshold here if necessary
        if tensor.numel() < 1000:
            print(f"{prefix}Values:\n{tensor}")
        else:
            print(f"{prefix}Values: [Tensor too large to display all values]")

    def print_data(data, indent=0):
        prefix = ' ' * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print_tensor_details(value, key, indent)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                print(f"{prefix}Index {i}:")
                print_data(item, indent + 2)
        elif hasattr(data, '__dict__'):  # Handle custom classes with attributes
            print(f"{prefix}Type: {type(data)}")
            for key, value in data.__dict__.items():
                if isinstance(value, torch.Tensor):
                    print_tensor_details(value, key, indent)
                else:
                    print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}Data: {data}")

    print("Type of the loaded data:", type(data))
    print_data(data)

# Execute the function
def view_pt():
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\subset\processed\processed\train_processed.pt'
    # file_path = r'C:\Users\BDY\Desktop\processed\processed\test_processed.pt'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\subset\processed\processed\train_processed.pt'

    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\triple\subset\processed\processed\train_processed.pt'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\subset\processed\processed\train_processed.pt'
    # file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\subset\processed\\train.pt'
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\PTC\subset\processed\processed\train_processed.pt'
    file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\subset\processed\processed\train_processed.pt'
    file_path = r'data\MUTAG\subset\processed\processed\train_processed.pt'

    # file_path = r'C:\Users\BDY\Desktop\processed\processed\train_processed.pt'
    data = torch.load(file_path)

    def print_tensor_details(tensor, name, indent):
        prefix = ' ' * indent
        print(f"{prefix}{name}: shape={tensor.shape}, dtype={tensor.dtype}")
        # Adjust the threshold here if necessary
        if tensor.numel() < 1000:
            print(f"{prefix}Values:\n{tensor}")
        else:
            print(f"{prefix}Values: [Tensor too large to display all values]")

    def print_data(data, indent=0, limit=10):
        prefix = ' ' * indent
        if isinstance(data, dict):
            # Only iterate over first 10 items if the dictionary has more than 10 items
            for i, (key, value) in enumerate(data.items()):
                if i >= limit:
                    print(f"{prefix}... {len(data)-limit} more items")
                    break
                if isinstance(value, torch.Tensor):
                    print_tensor_details(value, key, indent)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, (list, tuple)):
            # Only process the first 10 elements of lists or tuples
            for i, item in enumerate(data[:limit]):
                print(f"{prefix}Index {i}:")
                print_data(item, indent + 2)
            if len(data) > limit:
                print(f"{prefix}... {len(data)-limit} more items")
        elif hasattr(data, '__dict__'):
            print(f"{prefix}Type: {type(data)}")
            for i, (key, value) in enumerate(data.__dict__.items()):
                if i >= limit:
                    print(f"{prefix}... {len(data.__dict__)-limit} more items")
                    break
                if isinstance(value, torch.Tensor):
                    print_tensor_details(value, key, indent)
                else:
                    print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}Data: {data}")

    print("Type of the loaded data:", type(data))
    print_data(data)

import networkx as nx
import matplotlib.pyplot as plt

def load_and_transform_single_molecule(pickle_path, cfg):
    """
    Load the first molecule from a pickle file, print and visualize its details before and after applying
    SubgraphsTransform transformation.

    Args:
    - pickle_path: Path to the pickle file.
    - cfg: Configuration object to be supplied to SubgraphsTransform.

    Returns:
    - transformed_data: Transformed Data object.
    """
    # Load data from pickle file
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    first_molecule = dataset[0]

    # Build edge index
    num_atoms = first_molecule['num_atom']
    atom_types = torch.tensor(first_molecule['atom_type'])
    bond_types = first_molecule['bond_type']
    edge_index = []
    edge_attr = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if bond_types[i][j] > 0:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Because it's an undirected graph
                edge_attr.append(bond_types[i][j])
                edge_attr.append(bond_types[j][i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Build Data object
    data = Data(x=atom_types.unsqueeze(1).float(),  # Atom types as node features
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_atoms)

    # Print and visualize before transformation
    print("\nBefore transformation:")
    print(f"Number of nodes (atoms): {data.num_nodes}")
    print(f"Node features (atom types): {data.x}")
    print(f"Edge index: {data.edge_index}")
    if hasattr(data, 'edge_attr'):
        print(f"Edge attributes (bond types): {data.edge_attr}")

    G_before = nx.Graph()
    G_before.add_edges_from(data.edge_index.numpy().T)
    plt.figure(figsize=(8, 8))
    nx.draw(G_before, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
    plt.title("Before Transformation")
    plt.show()

    # Apply SubgraphsTransform
    transform = SubgraphsTransform(cfg)
    transformed_data = transform(data)  # 这里进行了变换

    # Print and visualize after transformation
    print("\nAfter transformation:")
    print(f"Number of nodes (atoms): {transformed_data.num_nodes}")
    # 根据需要添加更多打印信息
    G_after = nx.Graph()
    G_after.add_edges_from(transformed_data.edge_index.numpy().T)
    plt.figure(figsize=(8, 8))
    nx.draw(G_after, with_labels=True, node_color='lightgreen', node_size=700, edge_color='k')
    plt.title("After Transformation")
    plt.show()
    print("\nAfter transformation:")
    print(f"Number of nodes (atoms): {transformed_data.num_nodes}")

    # 检查并打印子图相关信息，如果存在
    if hasattr(transformed_data, 'subgraphs_nodes_mapper'):
        print(f"Subgraphs Nodes Mapper: {transformed_data.subgraphs_nodes_mapper}")
    if hasattr(transformed_data, 'subgraphs_edges_mapper'):
        print(f"Subgraphs Edges Mapper: {transformed_data.subgraphs_edges_mapper}")

    # 检查并打印位置编码信息，如果存在
    if hasattr(transformed_data, 'hop_indicator'):
        print(f"Hop Indicator: {transformed_data.hop_indicator}")
    if hasattr(transformed_data, 'cut_RWPE'):
        print(f"Cut RWPE: {transformed_data.cut_RWPE}")
    if hasattr(transformed_data, 'ego_RWPE'):
        print(f"Ego RWPE: {transformed_data.ego_RWPE}")
    if hasattr(transformed_data, 'global_RWPE'):
        print(f"Global RWPE: {transformed_data.global_RWPE}")

    print("\nDetailed information of transformed data:")
    for key, value in transformed_data.__dict__.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={value.shape}")
        else:
            print(f"{key}: {value}")

    return transformed_data

def review_model():
    global cfg  # 使用全局变量cfg
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    pickle_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\raw\train.pickle'
    transformed_data = load_and_transform_single_molecule(pickle_path, cfg)
    print(transformed_data)
    model_path = 'model_final.pt'  # 更新为您的模型文件路径
    model = torch.load(model_path)
    model.eval()  # 设置为评估模式

    # 加载测试数据集
    # 假设 create_dataset(cfg) 函数能够返回训练集、验证集和测试集
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    _, _, test_dataset = create_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_perf = test(test_loader, model)  # 确保test函数定义与之前一致

    # 打印测试结果
    print(f'Test performance: {test_perf:.4f}')


def convert_pt_to_pkl():
    # 从.pt文件中加载数据
    # 指定PT文件路径和目标PKL文件路径
    pt_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\processed\complete_dataset.pt'
    pkl_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\EXP\raw\converted_dataset.pkl'
    data_list = torch.load(pt_path)

    # 将数据保存为.pkl文件
    with open(pkl_path, 'wb') as file:
        pickle.dump(data_list, file)

    print(f"Data converted from PT to PKL and saved to: {pkl_path}")


import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch

import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_from_pkl():
    file_path = r'data\EXP\test.pickle'

    """
    从pkl文件中加载并可视化图数据。
    参数:
        file_path (str): pkl文件的路径。
    """
    # 加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 检查数据是否为预期的列表形式
    if not isinstance(data, list) or not data:
        print("数据格式不符合预期：预期为非空列表。")
        return

    # 使用列表中的第一个元素
    graph_data = data[0]

    # 创建图对象
    G = nx.Graph()

    # 处理节点
    num_nodes = len(graph_data['x'])
    G.add_nodes_from(range(num_nodes))

    # 处理边，edge_index是双列表形式
    edge_index = graph_data['edge_index']
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.t().tolist()
    elif isinstance(edge_index, list):
        edge_index = [list(pair) for pair in zip(*edge_index)]  # 转换成(pair)列表

    # 添加边到图中
    G.add_edges_from(edge_index)

    # 调整布局和可视化参数
    pos = nx.spring_layout(G)  # 为图生成一个布局
    node_color = 'skyblue'  # 节点颜色
    node_size = 700         # 节点大小
    font_size = 12          # 字体大小
    font_color = 'darkred'  # 字体颜色

    # 绘制图形
    nx.draw(G, pos, with_labels=False, node_color=node_color, edge_color='black',
            node_size=node_size, font_size=font_size, font_color=font_color)

    # 标注节点特征（假设'x'已经是标识符）
    labels = {i: str(graph_data['x'][i]) for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color=font_color)

    # 如果有'y'，可以额外添加信息显示
    # if 'y' in graph_data:
    #     plt.title(f"Graph Visualization, Label: {graph_data['y']}")
    # else:
    #     plt.title("Graph Visualization")

    plt.show()


def validate_model():
    global cfg
    config_path = 'train/configs/zinc.yaml'
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    cfg.processed_dir = './data/ZINC/subset/processed/processed'
    transform = SubgraphsTransform(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'ZINC_SAGNN_500.pt'
    # model_path = 'ZINC_MPNN_500.pt'
    # 设置验证集数据目录
    root = 'data/ZINC'
    transform = SubgraphsTransform(cfg)
    # val_dataset = ZINC(root, subset=True, split='val', transform=transform)
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
    val_loader = DataLoader(val_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 加载模型
    model = torch.load(model_path)
    model.to(device)

    # 设置模型为评估模式
    model.eval()
    total_mae = 0
    total_count = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            targets = data.y.to(device)
            outputs = model(data).squeeze()
            mae = l1_loss(outputs, targets, reduction='sum')
            total_mae += mae.item()
            total_count += data.num_graphs  # 计算图数量

    avg_mae = total_mae / total_count
    print(f"Validation MAE: {avg_mae:.4f}")
    return avg_mae

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
    model_path = 'ZINC_SAGNN_500.pt'
    model = torch.load(model_path).to(cfg.device)
    print("模型初始化完成。")

    # 设置优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

    # 初始化记录训练损失和性能指标的列表
    train_losses, train_accuracies, val_performances, test_performances = [], [], [], []
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
        # 打印和记录结果


# 使用示例：
# visualize_graph_from_pkl('path_to_your_file.pkl')
if __name__ == "__main__":
    # convert_pt_to_pkl( )
    # review_model()
    #test_create_dataset_and_model()
    # view_pt()
    # exp_pickle()
    # test_create_dataset()
    # visualize_graph_from_pkl() #可视化
    validate_model()#模型验证
    # test_create_dataset_and_model_val()


