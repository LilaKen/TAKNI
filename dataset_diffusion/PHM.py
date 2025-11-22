import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_diffusion.SequenceDatasets import dataset
from dataset_diffusion.sequence_aug import *
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

signal_size = 1024

# Case One
Case1 = ['helical 1', "helical 2", "helical 3", "helical 4", "helical 5", "helical 6"]

label1 = [i for i in range(6)]
# Case Two
Case2 = ['spur 1', "spur 2", "spur 3", "spur 4", "spur 5", "spur 6", "spur 7", "spur 8"]

label2 = [i for i in range(8)]

# working condition

WC = {0: "30hz" + "_" + "High" + "_1.txt",
      1: "35hz" + "_" + "High" + "_1.txt",
      2: "40hz" + "_" + "High" + "_1.txt",
      3: "45hz" + "_" + "High" + "_1.txt"}


# generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        state1 = WC[N[k]]  # WC[0] can be changed to different working states
        for i in range(len(Case1)):
            root1 = os.path.join(root, Case1[i], Case1[i] + "_" + state1)
            path1 = os.path.join(root1)
            data1, lab1 = data_load(path1, label=label1[i])
            data += data1
            lab += lab1
    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename, usecols=0)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        # if start==0:
        # print(filename)
        # print(fl[start:end])
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class PHM(object):
    num_classes = 6  # Case 1 have 6 labels; Case 2 have 9 lables
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype


    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            common_transforms = Compose([
                ToTensor(),
                # Normalize(self.normlizetype),
            ])


            list_data = get_files(self.data_dir, self.source_N)
            data_pd_source = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd_source, test_size=0.2, random_state=40, stratify=data_pd_source["label"])
            source_train = dataset(list_data=train_pd)
            source_val = dataset(list_data=val_pd)

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd_target = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd_target, test_size=0.2, random_state=40, stratify=data_pd_target["label"])
            target_train = dataset(list_data=train_pd)
            target_val = dataset(list_data=val_pd)
            # 应用转换到每个子数据集
            source_train.transform = common_transforms
            source_val.transform = common_transforms
            target_train.transform = common_transforms
            target_val.transform = common_transforms
            # 合并数据集
            combined_dataset = ConcatDataset([source_train, source_val, target_train, target_val])

            return combined_dataset

        else:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            a = np.resize(list_data[0], (1560, 1024))
            b = np.resize(list_data[1], (1560, 1))
            header_name = []
            for i in range(1024): header_name.append('d' + str(i))
            pd.DataFrame(a).to_csv('PHM_data.csv', index=False, header=header_name)
            pd.DataFrame(b).to_csv('PHM_label.csv', index=False, header=['label'])

            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val
