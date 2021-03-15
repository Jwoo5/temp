import csv
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

# All of datasets in physionet2021(CPSC, PTB, PTBXL, ...) integreted
class PhysionetDataset(Dataset):
    def __init__(self, path, coding = 'CMSMLC', valid = False):
        super().__init__()

        self.coding = coding
        self.valid = valid
        with open(path + 'manifest/' + 'train.tsv' if not valid else 'valid.tsv', mode = 'r') as file:
            self.dataset = list(csv.reader(file))
            self.path = self.dataset[0][0]
            self.dataset = self.dataset[1:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname = self.dataset[index][0][:-4]
        record = scipy.io.loadmat(self.path + "/" + fname) # expected (Leads, Frames)
        record = torch.tensor(record, dtype = torch.float)

        leads, frames = record.shape
        n_views = 2 # number of time segments

        
        if self.coding == 'CMSC':
            raise NotImplementedError
        elif self.coding == 'CMLC':
            raise NotImplementedError
        elif self.coding == 'CMSMLC':
            # TODO Build dataset for CMSMLC
            # reference to https://github1s.com/danikiyasseh/CLOCS/blob/HEAD/prepare_dataset.py
            # from 563 line
            record = record.transpose(0,1)
            record = record.view(leads, frames // n_views, -1) # (leads, frames/2, 2)

        else:
            raise NotImplementedError
            

        return record['val'], record['label']