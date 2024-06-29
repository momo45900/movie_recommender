import torch
from torch.utils.data import DataLoader, Dataset

class MovieLensDataset(Dataset):
    def __init__(self, df, mu):
        self.user_ids = torch.tensor(df.userId.values)
        self.movie_ids = torch.tensor(df.movie_idx.values)
        self.ratings = torch.tensor(df.rating.values - mu, dtype=torch.float32)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

def data_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
