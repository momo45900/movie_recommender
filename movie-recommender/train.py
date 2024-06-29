import torch
import pandas as pd
from module.model import Recommender
from module.setup import MovieLensDataset, data_loaders
from module.process import train_test_spliter, dataframe_process_create
from module.engine import train
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Load and prepare data

dataframe_process_create(raw_dir='data/raw/ratings.csv')

df = pd.read_csv('data/processed/edit_ratings.csv')



df_train, df_test, num_users, num_movies, mu = train_test_spliter(df)


# Initialize variables
K = 10  # latent dimensionality
epochs = 25
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create datasets and dataloaders
train_dataset = MovieLensDataset(df_train, mu)
test_dataset = MovieLensDataset(df_test, mu)
train_loader, test_loader = data_loaders(train_dataset, test_dataset, batch_size)

# Create the model
model = Recommender(num_users=num_users, num_movies=num_movies, embedding_dim=K).to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
train_losses, test_losses = train(model, train_loader, test_loader, optimizer, criterion, epochs, device)

# Plot training and validation loss
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plot training and validation MSE
plt.plot(train_losses, label="Train MSE")
plt.plot(test_losses, label="Test MSE")
plt.legend()
plt.title('MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()