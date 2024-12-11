import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Check and use GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 129996 # Words count obtained using word tokenizer in the LSTM model.
                    # The count of words in the vocabulary was too big (3,79,000) and the program was experiencing severe latency.
                    # Count of tokens using Tiktokenizer - 1,00,255, which thrown out of bound exception
SEQ_LEN = 64
BATCH_SIZE = 256
EMBEDDING_DIM = 100
NUM_CLASSES = 5
LEARNING_RATE = 0.001
MODEL_SAVE_FREQ = 10
MODEL_NUM = 9
EPOCHS = 50

df = pd.read_csv(r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Data\amazon_reviews_preprocessed.csv")
texts = df['Text'].tolist()
labels = df['Score'].tolist()
labels = [label - 1 for label in labels]

# Tokenization with tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode(text)
        # Truncate to max_length
        encoding = encoding[:self.max_length]
        # Pad with 0
        padding_length = self.max_length - len(encoding)
        encoding = encoding + [0] * padding_length
        return torch.tensor(encoding), torch.tensor(label)
        
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# DataLoaders
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model Definition (Simple Feedforward)
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, output_dim=NUM_CLASSES):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Mean pooling over the sequence
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Model initialization
#vocab_size = len(train_dataset)
model = SentimentModel(vocab_size=VOCAB_SIZE)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss_print = f"{running_loss/len(train_loader):.4f}"
        print(f"Epoch {epoch+1}, Loss: {loss_print}")
        if (epoch+1) % MODEL_SAVE_FREQ == 0:
            torch.save(model, f"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\model_pt_{MODEL_NUM}_epoch-{epoch+1}_loss-{loss_print}.pth")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1) 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model
evaluate_model(model, test_loader)
