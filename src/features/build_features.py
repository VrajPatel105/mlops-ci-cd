import pandas as pd
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer

import yaml

logger = logging.getLogger('feature_engineering')
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


def load_params(params_path: str) -> int:
    """Load max_features from the params yaml file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
        logger.debug('max_features retrieved: %s', max_features)
        return max_features
    except FileNotFoundError as e:
        logger.error('Params file not found: %s', e)
        raise
    except KeyError as e:
        logger.error('Missing key in params file: %s', e)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise


def load_data(train_path: str, test_path: str):
    """Load and fillna the processed train/test data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.debug('Processed data loaded successfully')
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty CSV file: %s', e)
        raise


def apply_bow(train_data, test_data, max_features):
    """Apply Bag of Words (CountVectorizer) to the data."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('Bag of Words applied and data transformed')
        return train_df, test_df
    except KeyError as e:
        logger.error('Missing column in dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(train_df, test_df, data_path: str):
    """Save the featurized train/test data to csv."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"))
        logger.debug('Featurized data saved to %s', data_path)
    except Exception as e:
        logger.error('Error saving featurized data: %s', e)
        raise


def main():
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = load_data('./data/processed/train_processed.csv',
                                           './data/processed/test_processed.csv')
        train_df, test_df = apply_bow(train_data, test_data, max_features)
        save_data(train_df, test_df, os.path.join("data", "features"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()