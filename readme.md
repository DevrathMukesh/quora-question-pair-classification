# Quora Question Pairs Duplicate Detection

## Overview
The objective of this project is to determine whether two questions posted on Quora have the same intent, i.e., are semantically similar. With many users asking similar questions on the platform, identifying duplicates can enhance the user experience by reducing redundant content and helping users find relevant answers faster.

The project involves preprocessing the question pairs, feature engineering, and exploring various machine learning and deep learning models to predict whether a pair of questions is a duplicate.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Selection](#model-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deep Learning Models](#deep-learning-models)
- [Results and Analysis](#results-and-analysis)
- [Usage](#usage)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
## Dataset
The dataset is from the Quora Question Pairs competition hosted on Kaggle. It contains the following columns:

- **id**: Unique identifier for each question pair.
- **qid1, qid2**: Unique identifiers for each question in the pair.
- **question1, question2**: The actual questions.
- **is_duplicate**: Label indicating if the questions are duplicates (1 for duplicates, 0 for non-duplicates).

### Data Statistics
The dataset contains approximately 400,000 question pairs, with roughly 37% labeled as duplicates. This class imbalance is addressed during the model training phase.

## Dataset
The dataset is from the Quora Question Pairs competition hosted on Kaggle. It contains the following columns:

- **id**: Unique identifier for each question pair.
- **qid1, qid2**: Unique identifiers for each question in the pair.
- **question1, question2**: The actual questions.
- **is_duplicate**: Label indicating if the questions are duplicates (1 for duplicates, 0 for non-duplicates).

### Data Statistics
The dataset contains approximately 400,000 question pairs, with roughly 37% labeled as duplicates. This class imbalance is addressed during the model training phase.

## Dependencies
The project requires the following libraries:

- **Core Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Natural Language Processing (NLP)**: `nltk`, `fuzzywuzzy`, `distance`, `beautifulsoup4`
- **Deep Learning**: `tensorflow`, `keras`
- **Visualization**: `plotly`

- **Install the dependencies**:
  ```bash
  pip install -r requirements.txt

## Data Preprocessing
Data preprocessing is a crucial step in this project and involves the following:

### Text Cleaning:
- Lowercasing all text, removing punctuation, and decontracting contractions (e.g., "I've" to "I have").
- Replacing special symbols like %, $, and @ with their textual equivalents.
- Removing any HTML tags using BeautifulSoup.

### Handling Missing Values:
- Missing questions are replaced with placeholder text ("missing").

### Tokenization and Stop Words Removal:
- Tokenizing questions into individual words.
- Removing common stop words (e.g., "the", "is", "in") that do not contribute to the meaning of the question.

### Stemming and Lemmatization (Optional):
- Reducing words to their base or root form for better similarity matching.

## Feature Engineering
Feature engineering is used to extract meaningful features from the text data. The features created include:

### Length-based Features:
- **Difference in character lengths** of the two questions.
- **Difference in the number of words**.

### Token-based Features:
- **Common words count**: Number of words appearing in both questions.
- **Total unique words count**: Sum of unique words in both questions.
- **Shared words ratio**: Ratio of common words to total unique words.

### Sequence-based Features:
- **Longest common substring** between the two questions.
- **Similarity ratios** using Levenshtein distance and other string similarity metrics from fuzzywuzzy.

### Bag of Words (BoW) and TF-IDF Vectors:
- Vectorizing the questions using BoW and TF-IDF to convert text data into numerical format.

### Advanced NLP Features:
- **Word Embeddings**: Pre-trained Word2Vec or GloVe embeddings are used to convert words into dense vectors.
- **Topic Modeling (optional)**: Latent Dirichlet Allocation (LDA) to identify common topics.


## Exploratory Data Analysis (EDA)
The EDA provides insights into the dataset and helps guide feature engineering. Some of the key analyses include:

- **Distribution of duplicate vs. non-duplicate questions**: Checking the imbalance in the dataset.
- **Character and word length distributions**: Understanding the typical question lengths.
- **Correlation between features**: Identifying which features are most predictive.


## Model Selection
A variety of machine learning models are used to evaluate the prediction task:

- **Logistic Regression**: A baseline model for binary classification.
- **Decision Trees and Random Forests**: Ensemble methods that work well with structured data.
- **XGBoost**: A powerful boosting algorithm that often achieves state-of-the-art performance in structured data competitions.


## Model Training and Evaluation
### Training-Testing Split:
The data is split into training and testing sets, with 80% for training and 20% for testing.

### Evaluation Metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision, Recall, F1-Score**: Important metrics to assess performance, especially given class imbalance.
- **Confusion Matrix**: Provides insight into the types of classification errors.


## Deep Learning Models
1. **LSTM (Long Short-Term Memory)**: 
   - LSTMs are used to capture the sequential nature of question text.
   - An embedding layer converts words to dense vectors, followed by LSTM layers that process the sequences.

2. **BiLSTM (Bidirectional LSTM)**: 
   - Processes input sequences in both forward and backward directions.
   - Helps to better capture context from both sides of a sequence.

3. **GRU (Gated Recurrent Unit)**: 
   - Similar to LSTM but with fewer parameters, potentially faster training.

4. **CNN + RNN Combination (Optional)**: 
   - Convolutional Neural Networks (CNN) extract local patterns in text, followed by RNN layers for sequential dependencies.

### Hyperparameter Tuning
Hyperparameters like learning rate, number of units, dropout rates, and optimizer type are tuned for optimal model performance using grid search or random search.

## Results and Analysis
- **Machine Learning Models**: Random Forest and XGBoost achieved accuracies close to 79%. XGBoost performed slightly better than other models due to its gradient boosting nature.
- **Deep Learning Models**: LSTM and BiLSTM achieved accuracies around 70-72%, indicating that with more data or fine-tuning, performance could improve. GRU and CNN + RNN models showed potential, but further hyperparameter tuning was needed. Word2Vec with Logistic Regression: Accuracy was lower than the other approaches, suggesting embeddings alone might not be enough for this task.

## Usage
- **Clone the repository**:
  ```bash
  git clone https://github.com/DevrathMukesh/
  quora-question-pair-classification.git
- **To run the App**:
  ```bash
  
  cd .\Streamlit\
  pip install -r requirements.txt
  streamlit run app.py
