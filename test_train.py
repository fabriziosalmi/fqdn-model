import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
from collections import Counter
import re
from urllib.parse import urlparse
import tldextract
from unittest.mock import patch, MagicMock
from io import StringIO


# Import the functions from your script (assuming it's named fqdn_classifier.py)
from train import (
    load_data, feature_engineering, train_model, evaluate_model, predict_fqdn,
    calculate_entropy, longest_consecutive_chars, levenshtein_distance,
    iterative_feature_selection, train_ensemble
)  # Replace 'fqdn_classifier'

# Mock console and related functions for testing output
class MockConsole:
    def __init__(self):
        self.outputs = []
        self.log = lambda *args, **kwargs: None   # Added for Progress compatibility

    def print(self, *args):
        self.outputs.append(" ".join(map(str, args)))  # Simplified output capture

    def status(self, *args, **kwargs):
        return self  # Return self to allow chaining

    def get_time(self):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

mock_console = MockConsole()


# Helper function to create a small dataset for testing
def create_test_data():
    data = {
        'fqdn': ['example.com', 'bad.example.net', 'google.com', 'malicious.domain.co.uk', 'another-bad-site.com'],
        'label': [0, 1, 0, 1, 1]
    }
    return pd.DataFrame(data)

# --- Test Cases ---

class TestFqdnClassifier(unittest.TestCase):

    def setUp(self):
        """Setup method to create test data and other necessary objects."""
        self.test_data = create_test_data()
        self.blacklist_file = "test_blacklist.csv"
        self.whitelist_file = "test_whitelist.csv"
        # Create dummy CSV files
        self.test_data[self.test_data['label'] == 1].to_csv(self.blacklist_file, index=False, header=False)
        self.test_data[self.test_data['label'] == 0].to_csv(self.whitelist_file, index=False, header=False)

        #Basic Vectorizer
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5))

    def tearDown(self):
        """Clean up any created files or resources after each test."""
        try:
            os.remove(self.blacklist_file)
            os.remove(self.whitelist_file)
        except FileNotFoundError:
            pass  # Ignore if files weren't created
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")

    @patch('train.console', mock_console)  # Use the class-level mock
    def test_load_data_success(self):
        """Test successful loading of data."""
        data = load_data(self.blacklist_file, self.whitelist_file, skip_errors=False)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 5)
        self.assertIn('label', data.columns)
        self.assertIn('fqdn', data.columns)
        self.assertTrue(all(data['fqdn'].str.islower()))  # Check lowercase conversion
        self.assertTrue(data['fqdn'].str.match(r'^[a-z0-9.-]+$').all()) # Check the regex filter


    @patch('train.console', mock_console)
    def test_load_data_file_not_found(self):
        """Test handling of FileNotFoundError."""
        with self.assertRaises(SystemExit) as cm:  # Expect SystemExit for cleaner testing
            load_data("nonexistent_file.csv", "another_nonexistent.csv", skip_errors=False)
        self.assertEqual(cm.exception.code, 1)  # Check for the correct exit code

    @patch('train.console', mock_console)
    def test_load_data_empty_file(self):
        """Test handling of empty CSV files."""
        # Create an empty file
        open("empty_file.csv", "w").close()
        with self.assertRaises(SystemExit) as cm:
            load_data("empty_file.csv", "empty_file.csv", skip_errors=False)
        self.assertEqual(cm.exception.code, 1)
        os.remove("empty_file.csv")

    @patch('train.console', mock_console)
    def test_load_data_parser_error(self):
        """Test handling of CSV parsing errors."""

        # Create malformed CSV content
        with open("malformed.csv", "w") as f:
            f.write("example.com\n")
            f.write("bad,data\n")  # Malformed line

        with self.assertRaises(SystemExit) as cm:
            load_data("malformed.csv", self.whitelist_file, skip_errors=False)
        self.assertEqual(cm.exception.code, 1)

        # Test with skip_errors=True
        data = load_data("malformed.csv", self.whitelist_file, skip_errors=True)
        self.assertEqual(len(data), 4)  # Should have skipped the bad line

        os.remove("malformed.csv")

    def test_feature_engineering(self):
      """Test the feature engineering function."""
      df = feature_engineering(self.test_data.copy())

      # Check for expected features (basic checks, extend as needed)
      self.assertIn('fqdn_length', df.columns)
      self.assertIn('num_dots', df.columns)
      self.assertIn('entropy', df.columns)
      self.assertIn('tld', df.columns)
      self.assertIn('sld', df.columns)
      self.assertIn('longest_consecutive_digit', df.columns)

      # Check a few specific values (using the test data defined above)
      self.assertEqual(df[df['fqdn'] == 'example.com']['fqdn_length'].iloc[0], 11)
      self.assertEqual(df[df['fqdn'] == 'example.com']['num_dots'].iloc[0], 1)

      #Test Levenshtein Distance
      self.assertEqual(df[df['fqdn'] == 'example.com']['edit_distance_to_com'].iloc[0], 0)
      self.assertEqual(df[df['fqdn'] == 'google.com']['edit_distance_to_google'].iloc[0], 0)

    @patch('train.console', mock_console)
    def test_train_model_random_forest(self):
        """Test training a Random Forest model."""

        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {
            'model__n_estimators': [10, 20],
            'model__max_depth': [None, 5]
        }

        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps['model'], RandomForestClassifier)


    @patch('train.console', mock_console)
    def test_train_model_smote(self):
        """Test training with SMOTE."""
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False, use_smote=True)
        self.assertIsInstance(model, ImbPipeline)
        self.assertIsInstance(model.named_steps['smote'], SMOTE)

    @patch('train.console', mock_console)
    def test_train_model_scaling(self):
        """Test training with scaling."""
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=True, use_quantile_transform=False)
        self.assertIsInstance(model.named_steps['preprocessor'].transformers_[1][1].named_steps['scaler'], StandardScaler)

    @patch('train.console', mock_console)
    def test_train_model_quantile_transform(self):
      """Test training with quantile transform."""
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)

      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
      model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=True)
      self.assertIsInstance(model.named_steps['preprocessor'].transformers_[1][1].named_steps['quantile'], QuantileTransformer)

    @patch('train.console', mock_console)
    def test_evaluate_model(self):
      """Test the evaluate_model function."""

      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)

      # Split data and train a model
      train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])

      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
      model, _, feature_names, best_params = train_model(train_data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)

      # Evaluate
      with patch('train.console', mock_console), patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
          evaluate_model(test_data, model, self.vectorizer, feature_names, scale_data=False, use_quantile_transform=False, model_name='random_forest', best_params=best_params, output_dir=".")

          #Basic Check for Metric Output
          self.assertTrue(any("Accuracy" in line for line in mock_console.outputs))

    @patch('train.console', mock_console)
    def test_predict_fqdn(self):
      """Test the predict_fqdn function."""

      # Use the trained model from the previous test
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      train_data, _ = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
      model, _, feature_names, _ = train_model(train_data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)

      # Predict
      with patch('train.console', mock_console):  # Redirect console output for testing
          predict_fqdn('example.com', model, self.vectorizer, feature_names, scale_data=False, use_quantile_transform=False)

      # Check the output (using mock_console.outputs)
      self.assertTrue(any("Prediction for [bold]example.com[/bold]" in line for line in mock_console.outputs))


    def test_calculate_entropy(self):
        self.assertAlmostEqual(calculate_entropy("aaaa"), 0.0)  # All same characters
        self.assertGreater(calculate_entropy("abcd"), 0.0)     # Different characters
        self.assertAlmostEqual(calculate_entropy(""), 0.0)       # Empty string

    def test_longest_consecutive_chars(self):
        self.assertEqual(longest_consecutive_chars("aaabbbccc", "abc"), 3)  # Basic test
        self.assertEqual(longest_consecutive_chars("aabbcc", "abc"), 2)      # Two consecutive
        self.assertEqual(longest_consecutive_chars("xyz", "abc"), 0)          # None found
        self.assertEqual(longest_consecutive_chars("", "abc"), 0)             # Empty String

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)  # Classic example
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)        # Identical strings
        self.assertEqual(levenshtein_distance("abc", ""), 3)           # One empty string
        self.assertEqual(levenshtein_distance("", "abc"), 3)           # One empty string
        self.assertEqual(levenshtein_distance("ab", "abc"), 1)          # Insertion
        self.assertEqual(levenshtein_distance("abc", "ac"), 1)          # Deletion
        self.assertEqual(levenshtein_distance("abc", "abd"), 1)          # Substitution
    @patch('train.console', mock_console)
    def test_iterative_feature_selection(self):
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      initial_features = [col for col in data.columns if col not in ['fqdn','label', 'tld','sld','subdomain']]
      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}

      best_features, best_score, _, _, _ = iterative_feature_selection(
          data, self.vectorizer, "random_forest", param_grid, False, False, False, initial_features, num_iterations=2
      )
      self.assertTrue(len(best_features) > 0)
      self.assertTrue(best_score >= 0.0)

    @patch('train.console', mock_console)
    def test_train_ensemble(self):
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      param_grids_ensemble = {
          'random_forest': {'model__n_estimators': [10], 'model__max_depth': [None]},
          'gradient_boosting': {'model__n_estimators': [10], 'model__learning_rate': [0.1]},
          'logistic_regression': {'model__C': [1.0]},
          'svm': {'model__C': [1.0], 'model__kernel': ['rbf']},
          'naive_bayes': {},
          'adaboost': {'model__n_estimators': [10], 'model__learning_rate': [1.0]}
      }

      model, _, _, _ = train_ensemble(data, self.vectorizer, False, False, False, param_grids_ensemble)
      self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()