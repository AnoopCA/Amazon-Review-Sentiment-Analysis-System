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
EPOCHS = 4
MODEL_SAVE_FREQ = 1
MODEL_NUM = 6

# Hyperparameter Search Space using hyperopt
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(0.1)),  # Loguniform distribution for learning rate
    'max_seq_length': scope.int(hp.quniform('max_seq_length', 64, 512, 16)), # Maximum sequence length
    'embedding_dim': scope.int(hp.quniform('embedding_dim', 64, 512, 16)), # Dimension of word embeddings
    'lstm_units': scope.int(hp.quniform('lstm_units', 1, 128, 1)), # Number of LSTM units
    'attention_units': scope.int(hp.quniform('attention_units', 1, 64, 1)), # Dimension of attention mechanism
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'batch_size': scope.int(hp.quniform('batch_size', 32, 512, 64)),
}

# Objective function for hyperopt
def objective(params):
    global train_loader, val_loader, model, criterion  # Use global variables where necessary

    # Update global parameters
    global BATCH_SIZE, EMBEDDING_DIM, LSTM_UNITS, ATTENTION_UNITS, DROPOUT_RATE

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

    # Update DataLoader with new batch size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define the model
    model = SentimentModel().to(device)
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,  # Number of evaluations
    trials=trials
)

print("Best Hyperparameters:", best_params)
