# Amazon Review Sentiment Analysis System

## Description

The **Amazon Review Sentiment Analysis System** is a data science project aimed at classifying Amazon product reviews into sentiment categories based on the text of the reviews. This project preprocesses the review data, extracts features, and trains a sentiment analysis model using a neural network. The goal is to predict sentiment scores (1-5) for product reviews, providing insights into customer sentiment.

## Features

- Preprocessing of raw Amazon reviews dataset, including text normalization and cleaning.
- Exploration of review text lengths, unique words, and special character analysis.
- Visualizations of text length distributions.
- Use of a multi-layer perceptron (MLP) model for sentiment classification.
- Model evaluation based on accuracy and loss metrics.
- This project also experiments with optimizing an LSTM model using the Hyperopt library for hyperparameter tuning.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - `pandas`: Data manipulation
  - `matplotlib`: Visualization
  - `torch`: Deep learning framework
  - `tiktoken`: Tokenization
  - `scikit-learn`: Model evaluation and data preprocessing
  - `hyperopt`: Hyperparameter optimization
- **Tools**: Jupyter Notebook, CUDA (for GPU acceleration)

## Data

The project uses the **Amazon Reviews Dataset**, which contains reviews and corresponding scores (1-5) for various products. The dataset is cleaned, preprocessed, and transformed into a format suitable for training a sentiment analysis model.

- **Dataset Source**: Amazon product reviews (CSV file)
- **Preprocessing Steps**: 
  - Removal of HTML tags and special characters
  - Tokenization and padding of text data
  - Text length analysis and distribution visualization

## Installation

Follow these steps to set up the project locally:

1. Clone this repository:
    ```bash
    git clone https://github.com/AnoopCA/Amazon-Review-Sentiment-Analysis-System.git
    cd amazon-review-sentiment-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the **amazon_reviews.csv** dataset and place it in the `Data` folder.

4. Ensure you have the necessary environment to run Jupyter Notebooks or Python scripts.

## Usage

To use the project, follow these steps:

1. **Preprocess the Data**: Run the `preprocess.ipynb` notebook to load, clean, and preprocess the dataset.

2. **Train the Model**: Run the `mlp_model.py` script to train the sentiment analysis model. The model will be trained using the preprocessed data.

3. **Evaluate the Model**: After training, the model will be evaluated on the test dataset, and the accuracy score will be printed.

### Example Usage:

```bash
python mlp_model.py
