import torch
import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 129996
SEQ_LEN = 64
EMBEDDING_DIM = 100
NUM_CLASSES = 5

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
        # Pad if necessary
        padding_length = self.max_length - len(encoding)
        encoding = encoding + [0] * padding_length  # Pad with zeros
        return torch.tensor(encoding), torch.tensor(label)
        
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

# Define the prediction function
class Sentiment_Predict:
    def __init__(self):
        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Load the saved model
        model_path = r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Models\model_pt_8_epoch-50_loss-1.0903.pth"
        self.model = torch.load(model_path, map_location=device).to(device)
        self.model.eval()

    def predict_score(self, text, max_length=SEQ_LEN):
        with torch.no_grad():
            # Tokenize and preprocess the input text
            encoding = self.tokenizer.encode(text)
            encoding = encoding[:max_length]  # Truncate to max_length
            padding_length = max_length - len(encoding)
            encoding = encoding + [0] * padding_length  # Pad with zeros
            input_tensor = torch.tensor([encoding]).to(device)  # Add batch dimension
            outputs = self.model(input_tensor)
            pred = torch.argmax(outputs, dim=1).item()  # Get the class with the highest probability
        return pred + 1 # Add 1 to match original label range

# Test text
#text = "This product is amazing and exceeded my expectations!"
#score = predict_score(text)
# Display predictions
#print(f"Text: {text}\nPredicted Score: {score}")
