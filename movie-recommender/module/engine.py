import torch

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0

    for user_ids, movie_ids, ratings in dataloader:
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(user_ids, movie_ids).squeeze()
        
       
        
        loss = loss_fn(predictions, ratings)
        
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for user_ids, movie_ids, ratings in dataloader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            predictions = model(user_ids, movie_ids).squeeze()
            loss = loss_fn(predictions, ratings)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(dataloader)
    return avg_test_loss

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss = test_step(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {test_loss}')

    return train_losses, test_losses
