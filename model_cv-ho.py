import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.nn.utils.rnn import pad_sequence
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
EPOCHS = 10
MODEL_SAVE_FREQ = 1
MODEL_NUM = 7
HYPEROPT_MAX_EVALS = 5

# Hyperparameter Search Space using hyperopt
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(0.0001)),  # Loguniform distribution for learning rate
    'max_seq_length': scope.int(hp.quniform('max_seq_length', 32, 128, 16)), # Maximum sequence length
    'embedding_dim': scope.int(hp.quniform('embedding_dim', 128, 384, 16)), # Dimension of word embeddings
    'lstm_units': scope.int(hp.quniform('lstm_units', 1, 96, 1)), # Number of LSTM units
    'attention_units': scope.int(hp.quniform('attention_units', 1, 64, 1)), # Dimension of attention mechanism
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'batch_size': scope.int(hp.quniform('batch_size', 32, 128, 64)),
}

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
    def __init__(self, attention_units, LSTM_UNITS):
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
    def __init__(self, MAX_WORDS, EMBEDDING_DIM, DROPOUT_RATE, ATTENTION_UNITS, LSTM_UNITS):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(MAX_WORDS, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_UNITS, num_layers=3, bidirectional=True, batch_first=True, dropout=DROPOUT_RATE)
        self.attention = AttentionLayer(ATTENTION_UNITS, LSTM_UNITS)
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

# Objective function for hyperopt
def objective(params):
    global train_loader, val_loader, model, criterion  # Use global variables where necessary

    # Update global parameters
    global LEARNING_RATE, MAX_SEQ_LENGTH, EMBEDDING_DIM, LSTM_UNITS, ATTENTION_UNITS, DROPOUT_RATE, BATCH_SIZE

    # Extract parameters from space
    LEARNING_RATE = params['learning_rate']
    MAX_SEQ_LENGTH = params['max_seq_length']
    EMBEDDING_DIM = params['embedding_dim']
    LSTM_UNITS = params['lstm_units']
    ATTENTION_UNITS = params['attention_units']
    DROPOUT_RATE = params['dropout_rate']
    BATCH_SIZE = params['batch_size']
        
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

    train_dataset = SentimentDataset(X_train, y_train)
    val_dataset = SentimentDataset(X_val, y_val)

    # Update DataLoader with new batch size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define the model
    model = SentimentModel(MAX_WORDS, EMBEDDING_DIM, DROPOUT_RATE, ATTENTION_UNITS, LSTM_UNITS).to(device)
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            loss.backward()
            optimizer.step()
    
    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss  # Minimize validation loss

# Hyperopt search
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=HYPEROPT_MAX_EVALS, trials=trials)

print("Best Hyperparameters:", best_params)

