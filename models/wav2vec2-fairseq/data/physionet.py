import csv
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

# All of datasets in physionet2021(CPSC, PTB, PTBXL, ...) integreted
class PhysionetDataset(Dataset):
    def __init__(self, path, valid = False):
        super().__init__()

        self.valid = valid
        with open(path + 'manifest/' + 'train.tsv' if not valid else 'valid.tsv', mode = 'r') as file:
            self.dataset = list(csv.reader(file))
            self.path = self.dataset[0][0]
            self.dataset = self.dataset[1:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname = self.dataset[index][0][:-4]
        record = scipy.io.loadmat(self.path + "/" + fname)

        return record['val'], record['label']