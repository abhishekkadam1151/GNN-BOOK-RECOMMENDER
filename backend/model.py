"""
Standalone model module for the GNN Book Recommender.
This can be used for testing / training the model independently.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
books_df = pd.read_csv(os.path.join(BASE_DIR, "books.csv"))
print(books_df.columns)

titles = books_df['title'].values
authors = books_df['author'].values
genres = books_df['genre'].values
ratings = books_df['user_rating'].values

# Encoding
title_encoder = LabelEncoder()
author_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

titles_encoded = title_encoder.fit_transform(titles)
authors_encoded = author_encoder.fit_transform(authors)
genres_encoded = genre_encoder.fit_transform(genres)

# Normalize ratings
scaler = MinMaxScaler()
ratings_norm = scaler.fit_transform(ratings.reshape(-1, 1)).squeeze()

# Node features
node_features = torch.tensor(
    list(zip(titles_encoded, authors_encoded, genres_encoded, ratings_norm)),
    dtype=torch.float
)

# Simple edges
edge_index = torch.tensor(
    [[i, i+1] for i in range(len(titles)-1)],
    dtype=torch.long
).t().contiguous()

data = Data(x=node_features, edge_index=edge_index)


# GNN Model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, len(titles))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


model = GNN()


# Recommendation function
def recommend(book_title, top_k=5):
    if book_title not in title_encoder.classes_:
        return ["Book not found"]

    idx = title_encoder.transform([book_title])[0]

    model.eval()
    with torch.no_grad():
        out = model(data)

    scores = out[idx]
    _, indices = torch.topk(scores, top_k)

    recs = title_encoder.inverse_transform(indices.numpy())
    return list(recs)


if __name__ == "__main__":
    print("Testing model...")
    result = recommend("1984")
    print("Recommendations for '1984':", result)