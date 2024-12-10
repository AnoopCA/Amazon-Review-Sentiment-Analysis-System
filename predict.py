import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.nn.utils.rnn import pad_sequence
import pickle
import time
import timeit
import nltk
from nltk.tokenize import word_tokenize
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope
nltk.download('punkt')

# Set device - Tensorflow is unable to access GPU, issue related to LSTM layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_WORDS = 129996 # Vocabulary size
NUM_CLASSES = 5  # Sentiment score range
EPOCHS = 3 #100
MODEL_SAVE_FREQ = 1
MODEL_NUM = 8

LEARNING_RATE = 0.00002
MAX_SEQ_LENGTH = 96 #350  # Maximum sequence length
EMBEDDING_DIM = 256 #128  # Dimension of word embeddings
LSTM_UNITS = 75 #128  # Number of LSTM units
ATTENTION_UNITS = 12 #64  # Dimension of attention mechanism
DROPOUT_RATE = 0.2 # Dropout rate
BATCH_SIZE = 64

# Dataset and DataLoader
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.y[idx].to(device)

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, attention_units):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(2 * LSTM_UNITS, attention_units)
        self.V = nn.Linear(attention_units, 1)

    def forward(self, hidden_states):
        scores = self.V(torch.tanh(self.W(hidden_states)))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector, attention_weights

# Model Definition
class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(MAX_WORDS, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_UNITS, num_layers=3, bidirectional=True, batch_first=True, dropout=DROPOUT_RATE)
        self.attention = AttentionLayer(ATTENTION_UNITS)
        self.fc = nn.Sequential(
            nn.Linear(2 * LSTM_UNITS, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output

def predict_sentiment(model, text, word_to_idx, max_seq_length, device):
    # Tokenize the input text
    tokens = word_tokenize(text.lower())
    
    # Convert tokens to indices
    sequence = [word_to_idx.get(token, 0) for token in tokens]
    
    # Pad the sequence
    if len(sequence) > max_seq_length:
        sequence = sequence[:max_seq_length]
    else:
        sequence = sequence + [0] * (max_seq_length - len(sequence))
    
    # Convert to tensor
    input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Make prediction
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

# Load the saved model (replace the path with your saved model file)
saved_model_path = r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\model_pt_7_epoch-20_loss-1.2482.pth"
loaded_model = torch.load(saved_model_path, map_location=device)

# Load word_to_idx from a pickle file
with open(r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

# Example usage
sample_text = """great taffy at a great price. there was a wide assortment of yummy taffy. delivery was very quick. if your a taffy lover, this is a deal."""
predicted_class = predict_sentiment(loaded_model, sample_text, word_to_idx, MAX_SEQ_LENGTH, device)

print(f"Predicted Sentiment Class: {predicted_class}")
