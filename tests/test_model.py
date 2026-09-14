# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "VrajPatel105"
        repo_name = "mlops-ci-cd"


        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_processed.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

def test_model_performance(self):
    # Assumes the last column is the label and the first column contains raw text.
    text_column = self.holdout_data.columns[0]
    label_column = self.holdout_data.columns[-1]

    X_holdout_text = self.holdout_data[text_column].fillna("").astype(str)
    y_holdout = self.holdout_data[label_column]

    # Convert raw text into the same 4,000 TF-IDF/count features
    # used when LogisticRegression was trained.
    X_holdout = self.vectorizer.transform(X_holdout_text)

    # Sparse matrix works directly with sklearn LogisticRegression.
    y_pred_new = self.new_model.predict(X_holdout)

    accuracy_new = accuracy_score(y_holdout, y_pred_new)
    precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
    recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
    f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)

    expected_accuracy = 0.40
    expected_precision = 0.40
    expected_recall = 0.40
    expected_f1 = 0.40

    self.assertGreaterEqual(
        accuracy_new,
        expected_accuracy,
        f"Accuracy should be at least {expected_accuracy}; got {accuracy_new:.4f}",
    )
    self.assertGreaterEqual(
        precision_new,
        expected_precision,
        f"Precision should be at least {expected_precision}; got {precision_new:.4f}",
    )
    self.assertGreaterEqual(
        recall_new,
        expected_recall,
        f"Recall should be at least {expected_recall}; got {recall_new:.4f}",
    )
    self.assertGreaterEqual(
        f1_new,
        expected_f1,
        f"F1 should be at least {expected_f1}; got {f1_new:.4f}",
    )
if __name__ == "__main__":
    unittest.main()