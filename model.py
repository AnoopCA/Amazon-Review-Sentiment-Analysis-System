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

# Load dataset
dataset = pd.read_csv(r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Data\amazon_reviews_preprocessed.csv")
texts = dataset['Text']
sentiment_scores = dataset['Score'] - 1  # Scores as 0 to 4

# Tokenization and preprocessing
word_to_idx = {}
idx_to_word = []
tokenized_texts = []

# Build vocabulary
for text in texts:
    tokens = word_tokenize(text.lower())
    tokenized_texts.append(tokens)
    for token in tokens:
        if token not in word_to_idx:
            word_to_idx[token] = len(word_to_idx)
            idx_to_word.append(token)

# Save word_to_idx to a pickle file
with open(r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)
print("word_to_idx saved!")

# Prepare data
sequences = [[word_to_idx.get(token, 0) for token in tokens] for tokens in tokenized_texts]
sequences = [torch.tensor(seq[:MAX_SEQ_LENGTH]) for seq in sequences]
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
padded_sequences = padded_sequences[:, :MAX_SEQ_LENGTH].to(device)

# One-hot encode labels
lb = LabelBinarizer()
y = torch.tensor(lb.fit_transform(sentiment_scores), dtype=torch.float32).to(device)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2)
X_train = X_train.to(device)
X_val = X_val.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

# Dataset and DataLoader
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.y[idx].to(device)

train_dataset = SentimentDataset(X_train, y_train)
val_dataset = SentimentDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

model = SentimentModel().to(device)

start_time = timeit.default_timer()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    loss_print = f"{train_loss/len(train_loader):.4f}"
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_print}")
    if (epoch+1) % MODEL_SAVE_FREQ == 0:
        torch.save(model, f"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\model_pt_{MODEL_NUM}_epoch-{epoch+1}_loss-{loss_print}.pth")

hours, minutes = divmod(int(timeit.default_timer() - start_time) // 60, 60)
print(f"Training time: {hours:02}:{minutes:02}")

# Validation
model.eval()
val_loss = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.argmax(dim=1))
        val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
