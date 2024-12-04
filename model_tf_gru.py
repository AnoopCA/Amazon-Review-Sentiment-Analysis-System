import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Bidirectional, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Hyperparameters
MAX_WORDS = 60  # Vocabulary size
MAX_SEQ_LENGTH = 35  # Maximum sequence length
EMBEDDING_DIM = 32  # Dimension of word embeddings
GRU_UNITS = 4  # Number of GRU units
ATTENTION_UNITS = 4  # Dimension of attention mechanism
DROPOUT_RATE = 0.5  # Dropout rate
NUM_CLASSES = 5  # Sentiment score range
BATCH_SIZE = 4
EPOCHS = 2

dataset = pd.read_csv(r"D:\ML_Projects\Amazon-Review-Sentiment-Analysis-System\Data\amazon_reviews_small.csv")
texts = dataset['Text']
sentiment_scores = dataset['Score'] - 1  # Scores as 0 to 4

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
y = tf.keras.utils.to_categorical(sentiment_scores, num_classes=NUM_CLASSES)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention_units):
        super(AttentionLayer, self).__init__()
        self.W = Dense(attention_units, activation='tanh')
        self.V = Dense(1)

    def call(self, hidden_states):
        score = self.V(self.W(hidden_states))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Model Definition
inputs = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH)(inputs)

# Stacked GRU layers
gru_output = x
for _ in range(3):  # Add more layers for deeper models
    gru_output = Bidirectional(GRU(GRU_UNITS, return_sequences=True, dropout=DROPOUT_RATE))(gru_output)

# Attention
attention_layer = AttentionLayer(ATTENTION_UNITS)
context_vector, attention_weights = attention_layer(gru_output)

# Fully connected layers
x = Dense(128, activation='relu')(context_vector)
x = Dropout(DROPOUT_RATE)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Compile model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)
