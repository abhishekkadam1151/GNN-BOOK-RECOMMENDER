from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Load dataset (resolve path relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
books_df = pd.read_csv(os.path.join(BASE_DIR, "books.csv"))

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

# Simple edges (connect consecutive books)
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
        return None  # Return None so caller can distinguish "not found"

    idx = title_encoder.transform([book_title])[0]

    model.eval()
    with torch.no_grad():
        out = model(data)

    scores = out[idx]
    _, indices = torch.topk(scores, min(top_k + 1, len(titles)))

    recs = title_encoder.inverse_transform(indices.numpy())
    # Exclude the input book itself from recommendations
    recs = [r for r in recs if r != book_title][:top_k]
    return recs


# Flask App
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests from frontend


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Book Recommender Backend Running ✅"})


@app.route("/recommend/<book>")
def get_recommendation(book):
    result = recommend(book)
    if result is None:
        return jsonify({"error": f"Book '{book}' not found in database"}), 404
    return jsonify(result)


@app.route("/books")
def list_books():
    """Return all available book titles (useful for the frontend)."""
    return jsonify(list(title_encoder.classes_))


if __name__ == "__main__":
    print("Book Recommender Backend starting...")
    print(f"Available books: {list(title_encoder.classes_)}")
    app.run(debug=True, port=5000)