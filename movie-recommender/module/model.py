import torch
from torch import nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        
        dot_product = torch.sum(user_embeds * movie_embeds, dim=1, keepdim=True)
        
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        
        prediction = dot_product.squeeze() + user_bias + movie_bias
        
        return prediction
