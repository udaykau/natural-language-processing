# Movie Review Sentiment Analysis with Wordlist and Naive Bayes Classifiers

This project explores Natural Language Processing (NLP) techniques for sentiment classification of movie reviews. It compares a simple wordlist-based classifier with a Naive Bayes classifier using the NLTK movie reviews corpus, and evaluates their performance using standard metrics.

## Features

- **Data Preparation:**  
  - Uses NLTK's movie reviews corpus.
  - Customizable train/test split for reproducible experiments.

- **Text Preprocessing:**  
  - Tokenization, normalization, and stopword removal.
  - Frequency analysis to extract representative positive and negative words.

- **Classification Approaches:**  
  - **Wordlist Classifier:** Classifies reviews based on the most frequent positive and negative words.
  - **Naive Bayes Classifier:** Utilizes NLTK’s probabilistic model for text classification.

- **Evaluation Metrics:**  
  - Accuracy, Precision, Recall, and F1 Score.
  - Custom confusion matrix implementation.

- **Experimentation:**  
  - Analyzes the impact of wordlist length on classifier performance.
  - Visualizes results using Matplotlib.

## Getting Started

### Prerequisites

- Python 3.6+
- Jupyter Notebook or Google Colab
- Required Python packages:
  - `nltk`
  - `pandas`
  - `matplotlib`

### Installation

1. Clone this repository or download the notebook file.
2. Install the required Python packages:
   ```sh
   pip install nltk pandas matplotlib
   ```
3. Download the necessary NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('movie_reviews')
   ```

### Usage

1. Open `ANLPassignment2023.ipynb` in Jupyter Notebook or Google Colab.
2. Set your candidate number in the designated cell for reproducible data splits.
3. Run all cells sequentially to:
   - Prepare and preprocess the data.
   - Generate word lists and train classifiers.
   - Evaluate and compare classifier performance.
   - Visualize experimental results.

## Results

- The Naive Bayes classifier consistently outperforms the wordlist classifier in terms of accuracy, precision, recall, and F1 score.
- Increasing the length of the wordlists improves the performance of the wordlist classifier, but it remains less robust than the Naive Bayes approach, especially on imbalanced datasets.

## Recommendation

For future work and real-world applications, the Naive Bayes classifier is recommended due to its probabilistic framework, ability to handle large feature sets, and superior performance on both balanced and imbalanced datasets.
