# 4. ml pipeline : model building

import os
import pandas as pd
import logging

from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import yaml

logger = logging.getLogger('model_building')
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


def load_data(train_path: str):
    """Load the featurized training data."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        logger.debug('Training data loaded, shape: %s', train_data.shape)
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty CSV file: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, output_dir: str):
    """Save the trained model to a pickle file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', output_dir)
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def main():
    try:
        X_train, y_train = load_data('./data/features/train_bow.csv')
        logistic_model = train_model(X_train, y_train)
        save_model(logistic_model, "models")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()