import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import train
train.MAX_JOBS = 1

# Import the functions from your script
from train import (
    load_data, feature_engineering, train_model, evaluate_model, predict_fqdn,
    calculate_entropy, longest_consecutive_chars, levenshtein_distance,
    iterative_feature_selection, train_ensemble
)

# Mock console and related functions for testing output
class MockConsole:
    def __init__(self):
        self.outputs = []
        self.log = lambda *args, **kwargs: None
    def print(self, *args):
        self.outputs.append(" ".join(map(str, args)))
    def status(self, *args, **kwargs):
        return self
    def get_time(self):
        return 0
    def set_live(self, live):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

mock_console = MockConsole()

# Dummy Progress context manager
class DummyProgress:
    def __init__(self, *args, **kwargs):
        pass
    def start(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, tb):
        pass
    def add_task(self, description, total=None):
        return self
    def update(self, task, advance=None, description=None): # Add this
        return self

# Helper function to create a *larger* dataset for testing
def create_test_data(num_samples=20):  # Increased sample size
    data = {
        'fqdn': [],
        'label': []
    }
    for i in range(num_samples):
        if i % 2 == 0:
            data['fqdn'].append(f'example{i}.com')
            data['label'].append(0)
        else:
            data['fqdn'].append(f'bad-example{i}.net')
            data['label'].append(1)
    return pd.DataFrame(data)

class TestFqdnClassifier(unittest.TestCase):

    def setUp(self):
        mock_console.outputs = []
        self.test_data = create_test_data() # Use the larger dataset
        self.blacklist_file = "test_blacklist.csv"
        self.whitelist_file = "test_whitelist.csv"

        self.test_data.loc[self.test_data['label'] == 1, 'fqdn'] = self.test_data.loc[self.test_data['label'] == 1, 'fqdn'].str.lower()
        self.test_data.loc[self.test_data['label'] == 0, 'fqdn'] = self.test_data.loc[self.test_data['label'] == 0, 'fqdn'].str.lower()
        self.test_data[self.test_data['label'] == 1].to_csv(self.blacklist_file, index=False, header=True)
        self.test_data[self.test_data['label'] == 0].to_csv(self.whitelist_file, index=False, header=True)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5))
        self.progress_patch = patch("train.Progress", DummyProgress)
        self.progress_patch.start()

        # Define a StratifiedKFold instance, but DON'T use it in the calls.
        self.cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    def tearDown(self):
        try:
            os.remove(self.blacklist_file)
            os.remove(self.whitelist_file)
        except FileNotFoundError:
            pass
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")
        self.progress_patch.stop()

    @patch('train.console', mock_console)
    def test_load_data_success(self):
        data = load_data(self.blacklist_file, self.whitelist_file, skip_errors=False)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 20)  # Corrected length. It should be 20
        self.assertIn('label', data.columns)
        self.assertIn('fqdn', data.columns)
        self.assertTrue(all(data['fqdn'].str.islower()))
        self.assertTrue(data['fqdn'].str.match(r'^[a-z0-9.-]+$').all())

    @patch('train.console', mock_console)
    def test_load_data_file_not_found(self):
        with self.assertRaises(SystemExit) as cm:
            load_data("nonexistent_file.csv", "another_nonexistent.csv", skip_errors=False)
        self.assertEqual(cm.exception.code, 1)

    @patch('train.console', mock_console)
    def test_load_data_empty_file(self):
        open("empty_file.csv", "w").close()
        with self.assertRaises(SystemExit) as cm:
            load_data("empty_file.csv", "empty_file.csv", skip_errors=False)
        self.assertEqual(cm.exception.code, 1)
        os.remove("empty_file.csv")

    @patch('train.console', mock_console)
    def test_load_data_parser_error(self):
        with open("malformed.csv", "w") as f:
            f.write("fqdn\n") #Include header.
            f.write("example.com\n")
            f.write("bad,data\n")
        with self.assertRaises(SystemExit) as cm:
            load_data("malformed.csv", self.whitelist_file, skip_errors=False)
        self.assertEqual(cm.exception.code, 1)
        data = load_data("malformed.csv", self.whitelist_file, skip_errors=True)
        self.assertEqual(len(data), 12)  # Corrected length: 10 from whitelist + 2 good from malformed
        os.remove("malformed.csv")

    def test_feature_engineering(self):
      df = feature_engineering(self.test_data.copy())
      self.assertIn('fqdn_length', df.columns)
      self.assertIn('num_dots', df.columns)
      self.assertIn('entropy', df.columns)
      self.assertIn('tld', df.columns)
      self.assertIn('sld', df.columns)
      self.assertIn('longest_consecutive_digit', df.columns)
      self.assertEqual(df[df['fqdn'] == 'example0.com']['fqdn_length'].iloc[0], 12)
      self.assertEqual(df[df['fqdn'] == 'example0.com']['num_dots'].iloc[0], 1)
      self.assertEqual(df[df['fqdn'] == 'example0.com']['edit_distance_to_com'].iloc[0], 0)
      self.assertEqual(df[df['fqdn'] == 'example0.com']['edit_distance_to_google'].iloc[0], 6)


    @patch('train.console', mock_console)
    def test_train_model_random_forest(self):
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {
            'model__n_estimators': [10, 20],
            'model__max_depth': [None, 5]
        }
        # REMOVE cv=self.cv
        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps['model'], RandomForestClassifier)

    @patch('train.console', mock_console)
    def test_train_model_smote(self):
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
        # REMOVE cv=self.cv
        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False, use_smote=True)
        self.assertIsInstance(model, ImbPipeline)
        self.assertIsInstance(model.named_steps['smote'], SMOTE)

    @patch('train.console', mock_console)
    def test_train_model_scaling(self):
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
        # REMOVE cv=self.cv
        model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=True, use_quantile_transform=False)
        self.assertIsInstance(model.named_steps['preprocessor'].transformers_[1][1].named_steps['scaler'], StandardScaler)

    @patch('train.console', mock_console)
    def test_train_model_quantile_transform(self):
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
      # REMOVE cv=self.cv
      model, _, _, _ = train_model(data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=True)
      self.assertIsInstance(model.named_steps['preprocessor'].transformers_[1][1].named_steps['quantile'], QuantileTransformer)

    @patch('train.console', mock_console)
    def test_evaluate_model(self):
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
      # REMOVE cv=self.cv
      model, _, feature_names, best_params = train_model(train_data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)
      with patch('train.console', mock_console), patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
          evaluate_model(test_data, model, self.vectorizer, feature_names, scale_data=False, use_quantile_transform=False, model_name='random_forest', best_params=best_params, output_dir=".")
          self.assertTrue(any("Accuracy" in line for line in mock_console.outputs) or any("accuracy" in line for line in mock_console.outputs))

    @patch('train.console', mock_console)
    def test_predict_fqdn(self):
        data = feature_engineering(self.test_data.copy())
        data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
        train_data, _ = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
        param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}
        # REMOVE cv=self.cv
        model, _, feature_names, _ = train_model(train_data, self.vectorizer, 'random_forest', param_grid, scale_data=False, use_quantile_transform=False)

        # Create a DataFrame for the input FQDN.  This is correct.
        input_df = pd.DataFrame({'fqdn': ['example.com']})
        predict_fqdn(input_df, model, self.vectorizer, feature_names, scale_data=False, use_quantile_transform=False)
        self.assertTrue(any("Prediction for [bold]example.com[/bold]" in line for line in mock_console.outputs))

    def test_calculate_entropy(self):
        self.assertAlmostEqual(calculate_entropy("aaaa"), 0.0)
        self.assertGreater(calculate_entropy("abcd"), 0.0)
        self.assertAlmostEqual(calculate_entropy(""), 0.0)

    def test_longest_consecutive_chars(self):
        self.assertEqual(longest_consecutive_chars("aaabbbccc", "abc"), 3)
        self.assertEqual(longest_consecutive_chars("aabbcc", "abc"), 2)
        self.assertEqual(longest_consecutive_chars("xyz", "abc"), 0)
        self.assertEqual(longest_consecutive_chars("", "abc"), 0)

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("ab", "abc"), 1)
        self.assertEqual(levenshtein_distance("abc", "ac"), 1)
        self.assertEqual(levenshtein_distance("abc", "abd"), 1)

    @patch('train.console', mock_console)
    def test_iterative_feature_selection(self):
      data = feature_engineering(self.test_data.copy())
      data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
      initial_features = [col for col in data.columns if col not in ['fqdn','label', 'tld','sld','subdomain']]
      param_grid = {'model__n_estimators': [10], 'model__max_depth': [None]}

      # Keep 'fqdn' for TF-IDF, but it won't be part of the iterative selection.
      X = data.drop('label', axis=1)  # Keep 'fqdn' here
      y = data['label']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

      # Fit the vectorizer on the training data's 'fqdn' column *before* dropping it.
      X_train_tfidf = self.vectorizer.fit_transform(X_train['fqdn']).toarray()
      X_test_tfidf = self.vectorizer.transform(X_test['fqdn']).toarray()

      # Convert the TF-IDF sparse matrices to DataFrames.
      tfidf_feature_names = self.vectorizer.get_feature_names_out()
      X_train_tfidf_df = pd.DataFrame(X_train_tfidf, columns=tfidf_feature_names, index=X_train.index)
      X_test_tfidf_df = pd.DataFrame(X_test_tfidf, columns=tfidf_feature_names, index=X_test.index)

      # Concatenate TF-IDF features with the other engineered features.
      X_train = pd.concat([X_train.drop(columns=['fqdn']), X_train_tfidf_df], axis=1)
      X_test = pd.concat([X_test.drop(columns=['fqdn']), X_test_tfidf_df], axis=1)
      # Create combined dataframes for use within iterative feature selection
      train_data = pd.concat([X_train, y_train], axis=1)
      test_data = pd.concat([X_test, y_test], axis=1)

        # REMOVE cv=self.cv
        # Now pass the modified dataframes, not X_train, y_train, etc.  Remove keyword args.
      best_features, best_score, _, _, _ = iterative_feature_selection(
          train_data, self.vectorizer, "random_forest", param_grid, False, False, False, initial_features, num_iterations=2,
          test_data=test_data
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

      # REMOVE cv=self.cv
      model, _, _, _ = train_ensemble(data, self.vectorizer, False, False, False, param_grids_ensemble)
      self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()