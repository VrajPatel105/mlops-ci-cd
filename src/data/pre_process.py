# ml pipeline 2. : data preprocessing
import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import logging

nltk.download('stopwords')
nltk.download('wordnet')

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error("Unexpected error during pre processing : %s", e)
        raise

def remove_stop_words(text):
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error("Error removing stop words: %s", e)
        raise

def removing_numbers(text):
    """Remove numbers from the text."""
    try:
        text = ''.join([char for char in text if not char.isdigit()])
        return text
    except Exception as e:
        logger.error("Error removing numbers: %s", e)
        raise

def lower_case(text):
    """Convert text to lower case."""
    try:
        text = text.split()
        text = [word.lower() for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error("Error converting to lower case: %s", e)
        raise

def removing_punctuations(text):
    """Remove punctuations from the text."""
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error("Error removing punctuations: %s", e)
        raise

def removing_urls(text):
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error("Error removing URLs: %s", e)
        raise

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully')

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"))
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('Empty CSV file: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()