import torch

# 设置文件路径
file_path = r'C:\Users\BDY\PycharmProjects\sagnn_createdataset\data\ZINC\subset\processed\processed\train_processed.pt'

# 加载文件
data_list = torch.load(file_path)

# 检查第一个元素的属性名称
if len(data_list) > 0:
    first_data = data_list[0]
    if hasattr(first_data, '_store'):
        print("Element 1:")
        for key, value in first_data._store.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: Tensor, Number of items {value.size()}")
            elif isinstance(value, int):
                print(f"    {key}: int")
            elif isinstance(value, list):
                print(f"    {key}: List, Number of items {len(value)}")
            else:
                print(f"    {key}: {type(value).__name__}")
    else:
        print("The object does not have '_store' attribute.")
else:
    print("The data list is empty.")
