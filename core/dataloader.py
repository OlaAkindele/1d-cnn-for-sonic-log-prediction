import torch
import numpy as np
from torch.utils.data import ( DataLoader, TensorDataset,Dataset)
from torch.utils.data.sampler import SubsetRandomSampler

def dataloader(x, y, bs=64):

    inputs = torch.from_numpy(x.values).cuda().float()
    outputs = torch.from_numpy(y.values).cuda().float()
    tensor = TensorDataset(inputs, outputs)
    loader = DataLoader(tensor, bs, shuffle=True, drop_last=True)

    return loader

#Buggy
#-----------------------------------------------------------------------------------------
class LogDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.values)
        self.y = torch.from_numpy(y.values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y

def dloader(x, y, bs=64, split='train', val_size=0.3):

    tensor = LogDataset(x, y)

    # obtain training indices that will be used for validation
    num_train = len(tensor)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_train))
    train_idx, val_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches

    if split=='train':
        loader = DataLoader(tensor, bs,
                            sampler=SubsetRandomSampler(train_idx),
                            num_workers=0)
        return loader

    elif split=='val':
        loader = DataLoader(tensor, bs,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=0)
        return loader
#--------------------------------------------------------------------------------------------------------------------