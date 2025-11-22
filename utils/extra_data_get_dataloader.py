from torch.utils.data import DataLoader
import numpy as np


# 从DataLoader中提取数据
def extract_data(dataloader):
    data_list, label_list = [], []
    for inputs, labels in dataloader:
        data_list.append(inputs.numpy())
        label_list.append(labels.numpy())
    data = np.vstack(data_list)
    labels = np.concatenate(label_list)
    return data, labels


def get_dataloader(args, Dataset):
    datasets = {}
    if isinstance(args.transfer_task[0], str):
        # print(args.transfer_task)
        args.transfer_task = eval("".join(args.transfer_task))

    # PU_type
    # datasets['source_train'], datasets['source_val'], datasets['target_train'], datasets[
    #     'target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

    datasets = Dataset(args.data_dir, args.transfer_task).data_split(transfer_learning=True)

    return datasets