import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
from tqdm import tqdm


class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pkl']  # Your raw file names

    @property
    def processed_file_names(self):
        return ['data.pt']  # Your processed file names

    def download(self):
        # Download to self.raw_dir
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            dataset = pickle.load(f)

        data_list = []
        for item in tqdm(dataset, desc='Processing'):
            # Assuming 'item' is a dictionary with appropriate fields
            node_features = torch.tensor(item['node_features'], dtype=torch.float)
            edge_index = torch.tensor(item['edge_index'], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(item['edge_attr'], dtype=torch.float)
            label = torch.tensor([item['label']], dtype=torch.long)

            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_custom_dataset(root_path):
    dataset = CustomDataset(root=root_path)
    return dataset
