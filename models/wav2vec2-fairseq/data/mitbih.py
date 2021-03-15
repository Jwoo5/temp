import csv
import numpy as np
import scipy.io
from torch.utils.data import Dataset

class MITBIHDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path + 'manifest/train.tsv', 'r', newline = '') as file:
            self.dataset = list(csv.reader(file))
            self.path = self.dataset[0][0]
            self.dataset = self.dataset[1:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname = self.dataset[index][0][:-4]
        record = scipy.io.loadmat(self.path + "/" + fname)
        record['val'] = record['val'].astype(np.float32)
        return record['val'], record['label'][0]