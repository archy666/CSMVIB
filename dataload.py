import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import scale

class ImportData(Dataset):
    def __init__(self, data_path):
        self.data = sio.loadmat(data_path)
        self.views = [scale(view.astype(np.float32)) for view in self.data['X'][0]]
        self.labels = np.squeeze(self.data['Y']) if np.min(self.data['Y']) == 0 else np.squeeze(self.data['Y'] - 1)
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        views = [view[idx] for view in self.views]
        label = self.labels[idx]
        return {'views': views, 'label': label}
