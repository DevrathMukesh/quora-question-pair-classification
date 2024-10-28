import re
import numpy as np
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance
import os
import logging

# Set logging level to display debug information
logging.basicConfig(level=logging.INFO)

# Directly specify the path for CountVectorizer model
cv_path = './Streamlit/models/CountVectorizer.pkl'  # Direct path to model file

# Check if the file exists
if os.path.exists(cv_path):
    logging.info(f"Loading CountVectorizer from: {cv_path}")  # Log the path

    # Load CountVectorizer
    try:
        with open(cv_path, 'rb') as file:
            cv = pickle.load(file)  # Assign the loaded CountVectorizer to 'cv'
        logging.info("CountVectorizer loaded successfully.")
        print("CountVectorizer model loaded successfully.")  # Print success message
    except Exception as e:
        logging.error(f"Failed to load CountVectorizer: {e}")
else:
    logging.error(f"{cv_path} not found. Please check the file path.")


def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1 & w2)

# Total words count
def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1) + len(w2)


# Token features function
def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    STOP_WORDS = set(stopwords.words("english"))
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = {word for word in q1_tokens if word not in STOP_WORDS}
    q2_words = {word for word in q2_tokens if word not in STOP_WORDS}

    q1_stops = {word for word in q1_tokens if word in STOP_WORDS}
    q2_stops = {word for word in q2_tokens if word in STOP_WORDS}

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

# Length features function
def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0

    return length_features

# Fuzzy features function
def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features



# Define the preprocessing function
def preprocess(q):
    # Convert to lowercase and strip whitespace
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the dataset.
    q = q.replace('[math]', '')

    # Replace large numbers with abbreviations (k, m, b)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting common English contractions
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "'cause": "because", "could've": "could have", "couldn't": "could not",
        "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
        "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did",
        "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would",
        "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
        "she'll've": "she will have", "she's": "she is", "should've": "should have",
        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
        "so's": "so as", "that'd": "that would", "that'd've": "that would have",
        "that's": "that is", "there'd": "there would", "there'd've": "there would have",
        "there's": "there is", "they'd": "they would", "they'd've": "they would have",
        "they'll": "they will", "they'll've": "they will have", "they're": "they are",
        "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
        "what'll've": "what will have", "what're": "what are", "what's": "what is",
        "what've": "what have", "when's": "when is", "when've": "when have",
        "where'd": "where did", "where's": "where is", "where've": "where have",
        "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
        "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
        "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
        "you'll've": "you will have", "you're": "you are", "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)

    # Further replacements for contractions
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags using BeautifulSoup
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

# Query point creation function
def query_point_creator(q1, q2):
    input_query = []

    # Preprocess both queries
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # Collect features
    input_query.append(len(q1))  # Length of q1
    input_query.append(len(q2))  # Length of q2
    input_query.append(len(q1.split()))  # Word count of q1
    input_query.append(len(q2.split()))  # Word count of q2

    input_query.append(test_common_words(q1, q2))  # Common words count
    input_query.append(test_total_words(q1, q2))  # Total words count
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))  # Common to total ratio

    # Token features
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # Length features
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # Fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # Bag-of-words features
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    # Combine all features into a single query point
    return np.hstack((np.array(input_query).reshape(1, -1), q1_bow, q2_bow))