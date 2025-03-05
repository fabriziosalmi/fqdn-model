import sys
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, classification_report, average_precision_score, brier_score_loss, matthews_corrcoef, precision_recall_curve, roc_curve
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import datetime
import argparse
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
from rich.table import Table
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# --- Data Loading and Preprocessing ---
console = Console()  # Initialize console here

def load_data(blacklist_file, whitelist_file, skip_errors=False):
    try:
        df_black = pd.read_csv(blacklist_file, header=None, names=["fqdn"],
                               on_bad_lines=('skip' if skip_errors else 'error'))
        df_black['label'] = 1
        df_white = pd.read_csv(whitelist_file, header=None, names=["fqdn"],
                               on_bad_lines=('skip' if skip_errors else 'error'))
        df_white['label'] = 0
        data = pd.concat([df_black, df_white], ignore_index=True)
        data = data[data['fqdn'] != 'fqdn']  # Remove duplicate header rows if any
    except Exception as e:
        console.print(f"Error loading CSV files: {e}")
        sys.exit(1)

    data['fqdn'] = data['fqdn'].str.lower()
    data = data[data['fqdn'].str.match(r'^[a-z0-9.-]+$')]
    if data.empty:
        console.print("Error: No valid data loaded.")
        sys.exit(1)
    # Ensure 'fqdn' is a string:
    data['fqdn'] = data['fqdn'].astype(str)
    return data

# --- Feature Engineering ---

def calculate_entropy(text):
    """Calculates the Shannon entropy of a string."""
    if not text:
        return 0
    length = len(text)
    seen = Counter(text.encode('utf-8'))
    entropy = 0
    for count in seen.values():
        p_x = float(count) / length
        if p_x > 0:
            entropy -= p_x * np.log2(p_x)
    return entropy

def longest_consecutive_chars(text, char_set):
    """Finds the length of the longest consecutive sequence of the same character from char_set."""
    if not text:
        return 0
    max_count = 0
    current_count = 0
    prev_char = None
    for char in text:
        if char in char_set:
            if char == prev_char:
                current_count += 1
            else:
                current_count = 1
            max_count = max(max_count, current_count)
            prev_char = char
        else:
            current_count = 0
            prev_char = None
    return max_count

def feature_engineering(df):
    """Generates features from FQDN strings, optimized for performance."""
    features = {}

    features['fqdn_length'] = df['fqdn'].apply(len)
    features['num_dots'] = df['fqdn'].apply(lambda x: x.count('.'))
    features['num_digits'] = df['fqdn'].apply(lambda x: sum(c.isdigit() for c in x))
    features['num_hyphens'] = df['fqdn'].apply(lambda x: x.count('-'))
    features['num_non_alphanumeric'] = df['fqdn'].apply(lambda x: sum(not c.isalnum() for c in x))
    features['entropy'] = df['fqdn'].apply(calculate_entropy)
    features['has_hyphen'] = df['fqdn'].apply(lambda x: 1 if '-' in x else 0)
    extracted = df['fqdn'].apply(tldextract.extract)
    features['tld'] = extracted.apply(lambda x: x.suffix)
    features['sld'] = extracted.apply(lambda x: x.domain)
    features['subdomain'] = extracted.apply(lambda x: x.subdomain)
    features['tld_length'] = features['tld'].apply(len)
    features['sld_length'] = features['sld'].apply(len)
    features['subdomain_length'] = features['subdomain'].apply(len)
    features['digit_ratio'] = features['num_digits'] / (features['fqdn_length'] + 1e-9)
    features['non_alphanumeric_ratio'] = features['num_non_alphanumeric'] / (features['fqdn_length'] + 1e-9)
    features['starts_with_digit'] = df['fqdn'].apply(lambda x: 1 if len(x) > 0 and x[0].isdigit() else 0)
    features['ends_with_digit'] = df['fqdn'].apply(lambda x: 1 if len(x) > 0 and x[-1].isdigit() else 0)

    def vowel_consonant_ratio(text):
        vowels = "aeiou"
        vowel_count = sum(1 for char in text if char in vowels)
        consonant_count = sum(1 for char in text if char.isalpha() and char not in vowels)
        return vowel_count / (consonant_count + 1e-9)

    features['vowel_consonant_ratio'] = df['fqdn'].apply(vowel_consonant_ratio)
    for char in "._-":
        features[f'count_{char}'] = df['fqdn'].apply(lambda x: x.count(char))

    features['longest_consecutive_digit'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x, "0123456789"))
    features['longest_consecutive_consonant'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x.lower(), "bcdfghjklmnpqrstvwxyz"))
    features['num_subdomains'] = features['subdomain'].apply(lambda x: len(x.split('.')) if x else 0)
    features['sld_entropy'] = features['sld'].apply(calculate_entropy)
    features['subdomain_entropy'] = features['subdomain'].apply(calculate_entropy)
    features['subdomain_ratio'] = features['subdomain_length'] / (features['fqdn_length'] + 1e-9)
    features['num_vowels'] = df['fqdn'].apply(lambda x: sum(1 for char in x if char in "aeiou"))
    features['longest_consecutive_vowel'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x.lower(), "aeiou"))
    features['unique_char_count'] = df['fqdn'].apply(lambda x: len(set(x)))
    features['has_www'] = df['fqdn'].apply(lambda x: 1 if "www." in x else 0)
    features['num_consonants'] = df['fqdn'].apply(lambda x: sum(1 for char in x if char in "bcdfghjklmnpqrstvwxyz"))
    features['consonant_ratio'] = features['num_consonants'] / (features['fqdn_length'] + 1e-9)
    features['unique_vowels'] = df['fqdn'].apply(lambda x: len(set(char for char in x if char in "aeiou")))
    features['unique_consonants'] = df['fqdn'].apply(lambda x: len(set(char for char in x if char in "bcdfghjklmnpqrstvwxyz")))
    features['digit_letter_ratio'] = df['fqdn'].apply(lambda x: sum(c.isdigit() for c in x) / (sum(c.isalpha() for c in x) + 1e-9))
    features['most_common_char_freq'] = df['fqdn'].apply(lambda x: max(Counter(x).values()))
    features['double_letter_count'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x)-1) if x[i]==x[i+1]))
    features['digit_sum'] = df['fqdn'].apply(lambda x: sum(int(char) for char in x if char.isdigit()))
    features['average_digit'] = df['fqdn'].apply(lambda x: (sum(int(char) for char in x if char.isdigit())) / (sum(c.isdigit() for c in x) + 1e-9))
    features['alpha_ratio'] = df['fqdn'].apply(lambda x: sum(c.isalpha() for c in x) / (len(x) + 1e-9))
    features['num_uppercase'] = df['fqdn'].apply(lambda x: sum(1 for c in x if c.isupper()))
    features['uppercase_ratio'] = df['fqdn'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-9))
    features['num_vowel_clusters'] = df['fqdn'].apply(lambda x: len([grp for grp in re.findall(r'[aeiou]+', x.lower()) if grp]))
    features['num_consonant_clusters'] = df['fqdn'].apply(lambda x: len([grp for grp in re.findall(r'[bcdfghjklmnpqrstvwxyz]+', x.lower()) if grp]))
    features['has_ip'] = df['fqdn'].apply(lambda x: 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', x) else 0)
    features['ratio_special_chars'] = df['fqdn'].apply(lambda x: sum(not c.isalnum() for c in x) / (len(x) + 1e-9))
    features['avg_token_length'] = df['fqdn'].apply(lambda x: np.mean([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['token_count'] = df['fqdn'].apply(lambda x: len(x.split('.')))
    features['dot_position_variance'] = df['fqdn'].apply(lambda x: np.var([i for i, c in enumerate(x) if c == '.']) / (len(x) + 1e-9) if '.' in x else 0)
    features['unique_alphanumeric'] = df['fqdn'].apply(lambda x: len(set(c for c in x if c.isalnum())))
    features['vowel_proportion'] = df['fqdn'].apply(lambda x: sum(c in "aeiou" for c in x.lower()) / (len(x) + 1e-9))
    features['consonant_proportion'] = df['fqdn'].apply(lambda x: sum(c.isalpha() and c.lower() not in "aeiou" for c in x) / (len(x) + 1e-9))
    features['unique_special_chars'] = df['fqdn'].apply(lambda x: len(set(c for c in x if not c.isalnum())))
    features['starts_with_www'] = df['fqdn'].apply(lambda x: 1 if x.startswith("www.") else 0)
    features['ends_with_com'] = df['fqdn'].apply(lambda x: 1 if x.endswith(".com") else 0)
    features['ends_with_org'] = df['fqdn'].apply(lambda x: 1 if x.endswith(".org") else 0)
    features['ends_with_net'] = df['fqdn'].apply(lambda x: 1 if x.endswith(".net") else 0)
    features['starts_with_letter'] = df['fqdn'].apply(lambda x: 1 if x and x[0].isalpha() else 0)
    features['ends_with_letter'] = df['fqdn'].apply(lambda x: 1 if x and x[-1].isalpha() else 0)
    features['tld_starts_with_vowel'] = features['tld'].apply(lambda x: 1 if x and x[0].lower() in "aeiou" else 0)
    features['sld_starts_with_vowel'] = features['sld'].apply(lambda x: 1 if x and x[0].lower() in "aeiou" else 0)
    features['subdomain_contains_www'] = features['subdomain'].apply(lambda x: 1 if "www" in x else 0)
    features['std_token_length'] = df['fqdn'].apply(lambda x: np.std([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['max_token_length'] = df['fqdn'].apply(lambda x: max([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['min_token_length'] = df['fqdn'].apply(lambda x: min([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['contains_numeric_only_token'] = df['fqdn'].apply(lambda x: 1 if any(token.isdigit() for token in x.split('.')) else 0)
    features['count_numeric_tokens'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if token.isdigit()))
    features['count_long_tokens'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) > 7))
    features['count_hyphen_tokens'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if '-' in token))
    features['vowel_to_consonant_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (x[i] in 'aeiou' and x[i+1] not in 'aeiou' and x[i+1].isalpha()) or (x[i] not in 'aeiou' and x[i].isalpha() and x[i+1] in 'aeiou')))
    features['consonant_to_vowel_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (x[i] not in 'aeiou' and x[i].isalpha() and x[i+1] in 'aeiou') or (x[i] in 'aeiou' and x[i+1] not in 'aeiou' and x[i+1].isalpha())))
    features['letter_to_digit_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (x[i].isalpha() and x[i+1].isdigit()) or (x[i].isdigit() and x[i+1].isalpha())))
    features['digit_to_letter_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (x[i].isdigit() and x[i+1].isalpha()) or (x[i].isalpha() and x[i+1].isdigit())))
    features['special_to_alphanum_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (x[i].isalnum() and not x[i+1].isalnum()) or (not x[i].isalnum() and x[i+1].isalnum())))
    features['alphanum_to_special_transitions'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if (not x[i].isalnum() and x[i+1].isalnum()) or (x[i].isalnum() and not x[i+1].isalnum())))
    features['vowel_density'] = df['fqdn'].apply(lambda x: sum(c in "aeiou" for c in x.lower()) / (sum(c.isalpha() for c in x.lower()) + 1e-9))
    features['consonant_density'] = df['fqdn'].apply(lambda x: sum(c.isalpha() and c.lower() not in "aeiou" for c in x) / (sum(c.isalpha() for c in x.lower()) + 1e-9))
    features['longest_vowel_cluster'] = df['fqdn'].apply(lambda x: max((len(cluster) for cluster in re.findall(r'[aeiou]+', x.lower())), default=0))
    features['longest_consonant_cluster'] = df['fqdn'].apply(lambda x: max((len(cluster) for cluster in re.findall(r'[bcdfghjklmnpqrstvwxyz]+', x.lower())), default=0))
    features['first_char_type'] = df['fqdn'].apply(lambda x: 1 if x[0].isdigit() else (2 if x[0].isalpha() else 3) if x else 0)
    features['last_char_type'] = df['fqdn'].apply(lambda x: 1 if x[-1].isdigit() else (2 if x[-1].isalpha() else 3) if x else 0)
    features['single_char_token_count'] =  df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) == 1))
    features['two_char_token_count'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) == 2))
    features['three_char_token_count'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) == 3))
    features['four_char_token_count'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) == 4))
    features['five_char_token_count'] = df['fqdn'].apply(lambda x: sum(1 for token in x.split('.') if len(token) == 5))
    features['tld_entropy'] = features['tld'].apply(calculate_entropy)
    features['sld_to_tld_length_ratio'] = features['sld_length'] / (features['tld_length'] + 1e-9)
    features['subdomain_to_sld_length_ratio'] = features['subdomain_length'] / (features['sld_length'] + 1e-9)
    features['vowel_count_in_sld'] = features['sld'].apply(lambda x: sum(1 for char in x if char in "aeiou"))
    features['consonant_count_in_sld'] = features['sld'].apply(lambda x: sum(1 for char in x if char in "bcdfghjklmnpqrstvwxyz"))
    features['digit_count_in_sld'] = features['sld'].apply(lambda x: sum(c.isdigit() for c in x))
    features['special_char_count_in_sld'] = features['sld'].apply(lambda x: sum(not c.isalnum() for c in x))
    features['vowel_count_in_subdomain'] = features['subdomain'].apply(lambda x: sum(1 for char in x if char in "aeiou"))
    features['consonant_count_in_subdomain'] = features['subdomain'].apply(lambda x: sum(1 for char in x if char in "bcdfghjklmnpqrstvwxyz"))
    features['digit_count_in_subdomain'] = features['subdomain'].apply(lambda x: sum(c.isdigit() for c in x))
    features['special_char_count_in_subdomain'] = features['subdomain'].apply(lambda x: sum(not c.isalnum() for c in x))

    # Corrected ratio calculations using np.where
    features['ratio_long_tokens'] = np.where(features['token_count'] > 0, features['count_long_tokens'] / features['token_count'], 0)
    features['unique_symbol_ratio'] = np.where(features['num_non_alphanumeric'] > 0, df['fqdn'].apply(lambda x: len(set(c for c in x if not c.isalnum()))) / features['num_non_alphanumeric'], 0)

    # --- Additional Features ---

    # Lexical Features based on character n-grams (bigrams and trigrams)
    for n in [2, 3]:
        features[f'{n}-gram_count'] = df['fqdn'].apply(lambda x: len([x[i:i+n] for i in range(len(x) - n + 1)]))
        features[f'unique_{n}-gram_count'] = df['fqdn'].apply(lambda x: len(set([x[i:i+n] for i in range(len(x) - n + 1)])))
        features[f'unique_{n}-gram_ratio'] = np.where(features[f'{n}-gram_count'] > 0, features[f'unique_{n}-gram_count'] / features[f'{n}-gram_count'], 0)

    # Features based on character repetition
    features['repeated_char_count'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x) - 1) if x[i] == x[i+1]))
    features['max_repeated_char_count'] = df['fqdn'].apply(lambda x: max(Counter(x).values()) if x else 0)


    # Features based on the presence of special keywords (beyond "www")
    keywords = ['login', 'account', 'secure', 'admin', 'mail', 'webmail', 'server', 'payment', 'bank', 'signin']
    for keyword in keywords:
        features[f'has_{keyword}'] = df['fqdn'].apply(lambda x: 1 if keyword in x else 0)


    # Position of first/last digit/letter
    features['first_digit_position'] = df['fqdn'].apply(lambda x: x.find(next((c for c in x if c.isdigit()), '')) if any(c.isdigit() for c in x) else -1)
    features['last_digit_position'] = df['fqdn'].apply(lambda x: len(x) - 1 - x[::-1].find(next((c for c in x[::-1] if c.isdigit()), '')) if any(c.isdigit() for c in x) else -1)
    features['first_letter_position'] = df['fqdn'].apply(lambda x: x.find(next((c for c in x if c.isalpha()), '')) if any(c.isalpha() for c in x) else -1)
    features['last_letter_position'] = df['fqdn'].apply(lambda x:  len(x) -1 - x[::-1].find(next((c for c in x[::-1] if c.isalpha()), '')) if any(c.isalpha() for c in x) else -1)

    # Number of different character types
    features['num_different_char_types'] = df['fqdn'].apply(lambda x: sum([any(c.isdigit() for c in x), any(c.isalpha() for c in x), any(not c.isalnum() for c in x)]))

    # Ratio of (vowels + digits) to length
    features['vowel_digit_ratio'] = df['fqdn'].apply(lambda x: (sum(c in "aeiou" for c in x.lower()) + sum(c.isdigit() for c in x) ) / (len(x) + 1e-9))

    # Create a new DataFrame from the features dictionary
    new_features_df = pd.DataFrame(features)

    # Concatenate the original DataFrame with the new features DataFrame
    df = pd.concat([df, new_features_df], axis=1)

    # Edit distance features (Levenshtein distance) – NOW calculated AFTER concatenation
    common_tlds = ['com', 'net', 'org', 'info', 'biz']
    for tld in common_tlds:
        df[f'edit_distance_to_{tld}'] = df['tld'].apply(lambda x: levenshtein_distance(x, tld) if pd.notna(x) else len(tld))

    common_slds = ['google', 'facebook', 'amazon', 'microsoft', 'apple']  # Example common SLDs
    for sld in common_slds:
        df[f'edit_distance_to_{sld}'] = df['sld'].apply(lambda x: levenshtein_distance(x, sld) if pd.notna(x) else len(sld))

    return df

def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def evaluate_model(test_data, model, vectorizer, feature_names, scale_data, use_quantile_transform, model_name, best_params, output_dir):
    """Evaluates the model (using the pipeline) and prints results."""

    # Separate features and target
    y_test = test_data['label']
    X_test_text = test_data['fqdn']
    X_test_engineered = test_data.drop(['fqdn', 'label', 'sld', 'subdomain'], axis=1)
    X_test_engineered = X_test_engineered.apply(pd.to_numeric, errors='coerce').fillna(0)


    # Combine text and engineered features for the pipeline
    X_test = pd.concat([X_test_text, X_test_engineered], axis=1)
    
    # Make predictions using the fitted pipeline
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_prob)
    brier_score = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    # --- Output Results (Rich Table) ---
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Accuracy", f"{accuracy:.4f}")
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("F1-Score", f"{f1:.4f}")
    table.add_row("AUC", f"{auc:.4f}")
    table.add_row("Log Loss", f"{logloss:.4f}")
    table.add_row("Average Precision", f"{ap_score:.4f}")
    table.add_row("Brier Score", f"{brier_score:.4f}")
    table.add_row("MCC", f"{mcc:.4f}")
    console.print(table)

    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(conf_matrix)

    console.print("\n[bold]Classification Report:[/bold]")
    console.print(report)

    # --- Model Settings and Hyperparameters ---
    console.print("\n[bold]Model Settings:[/bold]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Best Hyperparameters: {best_params}")
    console.print(f"  Scaling: {scale_data}")
    console.print(f"  Quantile Transform: {use_quantile_transform}")
    console.print(f"  Vectorizer: {type(vectorizer).__name__}")
    console.print(f"    N-gram Range: {vectorizer.ngram_range}")
    console.print(f"    Analyzer: {vectorizer.analyzer}")

    # --- Top 10 Feature Importances ---

    if hasattr(model, 'named_steps') and "model" in model.named_steps:
      if hasattr(model.named_steps["model"], 'feature_importances_'):
        try:
          feature_importances = model.named_steps["model"].feature_importances_
          fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
          fi_df = fi_df.sort_values(by='Importance', ascending=False)

          console.print("\n[bold]Top 10 Feature Importances:[/bold]")
          table = Table(show_header=True, header_style="bold cyan")
          table.add_column("Feature")
          table.add_column("Importance", justify="right")
          for _, row in fi_df.head(10).iterrows():
              table.add_row(row['Feature'], f"{row['Importance']:.4f}")
          console.print(table)
        except Exception as e:
            console.print(f"Could not determine feature importances: {e}")
    elif hasattr(model, 'feature_importances_'):
      #For cases where the model is not in a pipeline
      try:
        feature_importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        console.print("\n[bold]Top 10 Feature Importances:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Feature")
        table.add_column("Importance", justify="right")
        for _, row in fi_df.head(10).iterrows():
            table.add_row(row['Feature'], f"{row['Importance']:.4f}")
        console.print(table)
      except Exception as e:
        console.print("Could not determine feature importances")
    # --- Visualizations ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

    if hasattr(model, 'named_steps') and "model" in model.named_steps:
      if hasattr(model.named_steps["model"], 'feature_importances_'):
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), orient='h')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png")
        plt.close()
    elif hasattr(model, 'feature_importances_'):
      plt.figure(figsize=(8, 6))
      sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), orient='h')
      plt.title('Top 10 Feature Importances')
      plt.tight_layout()
      plt.savefig(f"{output_dir}/feature_importance.png")
      plt.close()


def predict_fqdn(fqdn, model, vectorizer, feature_names, scale_data, use_quantile_transform):
    """Predicts and displays results using Rich."""
    df = pd.DataFrame({'fqdn': [fqdn]})
    df = feature_engineering(df)
    df = pd.get_dummies(df, columns=['tld'], prefix='tld', dummy_na=False)

    # Reindex *before* combining with text features.
    df = df.reindex(columns=[col for col in feature_names if col in df.columns], fill_value=0)

    # Transform the FQDN using the vectorizer
    fqdn_tfidf = vectorizer.transform(df['fqdn'])

    # Get text feature names from the vectorizer
    text_feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame for the text features
    tfidf_df = pd.DataFrame(fqdn_tfidf.toarray(), columns=text_feature_names)

    # Combine text and engineered features
    X_unseen = pd.concat([
        tfidf_df,
        df.drop(['fqdn', 'sld', 'subdomain'], axis=1).reset_index(drop=True)
    ], axis=1)
    X_unseen = X_unseen.apply(pd.to_numeric, errors='coerce').fillna(0)  # Apply to numeric

    # Ensure all columns are present and in the correct order
    X_unseen = X_unseen.reindex(columns=feature_names, fill_value=0)
    X_unseen.columns = X_unseen.columns.astype(str)

    prediction = model.predict(X_unseen)[0]
    probability = model.predict_proba(X_unseen)[0, 1]

    color = "green" if prediction == 0 else "red"
    console.print(f"Prediction for [bold]{fqdn}[/bold]: [{color}]{'Benign' if prediction == 0 else 'Malicious'}[/{color}] (Probability: {probability:.4f})")



def train_ensemble(train_data, vectorizer, scale_data, use_quantile_transform, use_smote, param_grids):
    """Trains a VotingClassifier ensemble."""

    y_train = train_data['label']
    X_train_text = train_data['fqdn']
    X_train_engineered = train_data.drop(['fqdn', 'label', 'sld', 'subdomain'], axis=1, errors='ignore')
    X_train_engineered = X_train_engineered.apply(pd.to_numeric, errors='coerce').fillna(0) # Apply to numeric


    # --- Preprocessing Pipelines (as before) ---
    text_transformer = Pipeline([
        ('tfidf', vectorizer),
    ])

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler() if scale_data else 'passthrough'),
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42,
                                         n_quantiles=min(1000, X_train_engineered.shape[0])) if use_quantile_transform else 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'fqdn'),
            ('num', numeric_transformer, X_train_engineered.columns.tolist())
        ],
        remainder='passthrough'
    )
    # --- Define Base Models with Pipelines ---
    rf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),  # Conditional SMOTE
        ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=MAX_JOBS, class_weight='balanced'))),
        ('model', RandomForestClassifier(random_state=42, n_jobs=MAX_JOBS, class_weight='balanced'))
    ])

    gb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    lr_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),
        ('model', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
    ])

    svm_pipeline = ImbPipeline([
      ('preprocessor', preprocessor),
      ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),
      ('model', SVC(probability=True, random_state=42, class_weight='balanced')) #Need probability for VotingClassifier
    ])

    nb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),
        ('model', GaussianNB())
    ])

    ab_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42) if use_smote else 'passthrough'),
        ('model', AdaBoostClassifier(random_state=42))
    ])



    # --- Hyperparameter Tuning (RandomizedSearchCV) for Each Base Model ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def tune_model(pipeline, param_grid, model_name):
      with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
          task = progress.add_task(f"Tuning {model_name}...", total=None)
          random_search = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=MAX_JOBS, verbose=1, n_iter=7, random_state=42)
          #Fit on combined data
          random_search.fit(pd.concat([X_train_text, X_train_engineered], axis=1), y_train)

          progress.update(task, advance=1, description=f"Tuning {model_name}...[green]Done![/green]")
          return random_search.best_estimator_

    best_rf = tune_model(rf_pipeline, param_grids['random_forest'], 'RandomForest')
    best_gb = tune_model(gb_pipeline, param_grids['gradient_boosting'], 'GradientBoosting')
    best_lr = tune_model(lr_pipeline, param_grids['logistic_regression'], 'LogisticRegression')
    best_svm = tune_model(svm_pipeline, param_grids['svm'], 'SVM')
    best_nb = tune_model(nb_pipeline, param_grids['naive_bayes'], 'NaiveBayes')
    best_ab = tune_model(ab_pipeline, param_grids['adaboost'], 'AdaBoost')

    # --- Create Voting Classifier ---
    # Use named estimators from the tuned pipelines
    estimators = [
      ('rf', best_rf),
      ('gb', best_gb),
      ('lr', best_lr),
      ('svm', best_svm),
      ('nb', best_nb),
      ('ab', best_ab)
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=MAX_JOBS) #soft voting

    # --- Train Voting Classifier ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Training Voting Classifier...", total=None)

        voting_clf.fit(pd.concat([X_train_text, X_train_engineered], axis=1), y_train)

        progress.update(task, advance=1, description="Training Voting Classifier...[green]Done![/green]")


    # --- Get Feature Names (after preprocessing and selection) ---
    #   (This is tricky with nested pipelines; we'll get an approximation)
    try:
      feature_names = best_rf.named_steps['preprocessor'].get_feature_names_out()
      selected_features = best_rf.named_steps['feature_selection'].get_support()
      final_feature_names = feature_names[selected_features]
    except:
      final_feature_names = X_train_engineered.columns


    return voting_clf, vectorizer, final_feature_names, {}  # No single best_params for VotingClassifier


def iterative_feature_selection(train_data, vectorizer, model_name, param_grid, scale_data, use_quantile_transform, use_smote, initial_features, num_iterations=5, scoring='roc_auc'):
    """Performs iterative feature selection."""

    best_features = initial_features
    best_score = 0.0

    for iteration in range(num_iterations):
        console.print(f"\n[bold]Iteration {iteration + 1}/{num_iterations}[/bold]")

        # Prepare data with current feature set
        current_train_data = train_data[best_features + ['fqdn','label']]
        # Crucially:  Apply numeric conversion *only* to the engineered features.
        engineered_cols = [col for col in current_train_data.columns if col not in ['fqdn', 'label']]
        current_train_data.loc[:, engineered_cols] = current_train_data[engineered_cols].apply(pd.to_numeric, errors='coerce').fillna(0)



        # Train and evaluate model
        if model_name != "ensemble":
          model, _, feature_names, best_params = train_model(
            current_train_data, vectorizer, model_name, param_grid,
            scale_data, use_quantile_transform, use_smote
          )
        else:
          model,_,feature_names, best_params = train_ensemble(
            current_train_data, vectorizer, scale_data, use_quantile_transform, use_smote, param_grid
          )

        # Get feature importances (handle pipelines and non-pipeline models)
        if hasattr(model, 'named_steps') and "model" in model.named_steps:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
            else:
                console.print("Model does not have feature importances.")
                return best_features, best_score, model, feature_names, best_params
        elif hasattr(model, "feature_importances_"):
          importances = model.feature_importances_
        else:
            console.print("Model does not have feature importances.")
            return best_features, best_score, model, feature_names, best_params

        # Create DataFrame for feature importances
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        # Evaluate current model
        y_true = current_train_data['label']
        X_current = current_train_data.drop(['fqdn', 'label'], axis=1)
        y_pred_proba = model.predict_proba(X_current)[:, 1] # Use predict_proba
        current_score = roc_auc_score(y_true, y_pred_proba)
        console.print(f"Current AUC: {current_score:.4f}")

        #Update if the current iteration gives better results.
        if current_score > best_score:
          best_score = current_score
          best_features = list(fi_df['Feature'])
          best_model = model
          best_feature_names = feature_names
          best_best_params = best_params


        # Remove the least important feature
        if len(best_features) > 1: # Prevent removing all features.
          best_features = list(fi_df['Feature'])[:-1] # Remove last
        console.print(f"Selected Features: {len(best_features)}")

    console.print(f"[green]Best AUC after feature selection: {best_score:.4f}[/green]")
    return best_features, best_score, best_model, best_feature_names, best_best_params
def train_model(train_data, vectorizer, model_name, param_grid, scale_data, use_quantile_transform, use_smote=False):
    """Trains the model with RandomizedSearchCV and optional SMOTE."""

    # Separate features and target
    y_train = train_data['label']
    X_train_text = train_data['fqdn']
    X_train_engineered = train_data.drop(['fqdn', 'label', 'sld', 'subdomain'], axis=1, errors='ignore')
    X_train_engineered = X_train_engineered.apply(pd.to_numeric, errors='coerce').fillna(0) # Apply to numeric

    # Preprocessing for text features (TF-IDF)
    text_transformer = Pipeline([
        ('tfidf', vectorizer),
    ])

    # Preprocessing for numerical features (scaling and quantile transform)
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler() if scale_data else 'passthrough'),  # 'passthrough' if not scaling
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(1000, X_train_engineered.shape[0])) if use_quantile_transform else 'passthrough')
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'fqdn'),  # Apply text_transformer to 'fqdn' column
            ('num', numeric_transformer, X_train_engineered.columns.tolist())  # Apply numeric_transformer to other columns
        ],
        remainder='passthrough'  # Keep any remaining columns (shouldn't be any)
    )


    # Define the model
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=MAX_JOBS, class_weight='balanced')
    elif model_name == 'gradient_boosting':
      model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'logistic_regression':
      model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    elif model_name == 'svm':
      model = SVC(probability=True, random_state=42, class_weight='balanced')
    elif model_name == 'naive_bayes':
      model = GaussianNB()
    elif model_name == 'adaboost':
      model = AdaBoostClassifier(random_state=42)
    # ... (add other models here) ...
    else:
        raise ValueError(f"Invalid model name: {model_name}")


    # Create the pipeline (with or without SMOTE)
    if use_smote:
      pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)), #Add SMOTE for oversampling
            ('model', model)
        ])
    else:
      pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=MAX_JOBS, verbose=0, n_iter=9, random_state=42)  # n_iter controls the number of parameter settings sampled

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Tuning {model_name}...", total=None)
        # Fit on the combined data (ColumnTransformer handles the splitting)
        random_search.fit(pd.concat([X_train_text, X_train_engineered], axis=1), y_train)
        progress.update(task, advance=1, description=f"Tuning {model_name}...[green]Done![/green]")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Get feature names *after* preprocessing and feature selection

    try:
      feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

    except:
      feature_names =  X_train_engineered.columns # Fallback

    return best_model, vectorizer, feature_names, best_params

# --- Parameter Grids (defined outside main) ---
param_grids = {
'random_forest': {
    'model__n_estimators': [100, 150, 200],  # Example values
    'model__max_depth': [None, 10, 20],
},
'gradient_boosting': {
    'model__n_estimators': [100, 150, 200],
    'model__learning_rate': [0.1, 0.05, 0.2]
},
'logistic_regression': {
    'model__C': [1.0, 0.5, 2.0]
},
'svm': {
    'model__C': [1.0, 0.5, 2.0],
    'model__kernel': ['rbf', 'linear']
},
'naive_bayes': {},
'adaboost': {
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [1.0, 0.5, 2.0]
},
# Note: 'ensemble' key is NOT here; it's handled separately
}

# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="FQDN Classifier", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("blacklist", help="Path to the blacklist file")
    parser.add_argument("whitelist", help="Path to the whitelist file")
    parser.add_argument("-p", "--predict", help="FQDN to predict", default=None)
    parser.add_argument("-ts", "--test_size", help="Test size (default: 0.3)", type=float, default=0.3)
    parser.add_argument("-rs", "--random_state", help="Random state (default: 42)", type=int, default=42)
    parser.add_argument("-m", "--model", help="Model to use (default: random_forest)\n"
                                               "Available models:\n"
                                               "  random_forest\n"
                                               "  gradient_boosting\n"
                                               "  logistic_regression\n"
                                               "  svm\n"
                                               "  naive_bayes\n"
                                               "  adaboost\n"
                                               "  ensemble (VotingClassifier)",
                        default="random_forest")
    parser.add_argument("-ng", "--ngram_range", help="N-gram range (default: 2 4)", nargs=2, type=int, default=[2, 4])
    parser.add_argument("--skip_errors", action="store_true", help="Skip malformed lines")
    parser.add_argument("--scale", action="store_true", help="Scale features")
    parser.add_argument("--quantile_transform", action="store_true", help="Apply quantile transformation")
    parser.add_argument("-o", "--output_dir", help="Output directory (default: results)", default="results")
    parser.add_argument("-s","--save_model", help="Save the best model to a file (default: model/best_model.pkl)", default="model/best_model.pkl")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE for oversampling")
    parser.add_argument("--l1", action="store_true", help="Use L1 regularization for text features")  # L1 regularization
    parser.add_argument("--l2", action="store_true", help="Use L2 regularization for text features")  # L2 regularization
    parser.add_argument("--feature_selection", action="store_true", help="Perform iterative feature selection")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of iterations for feature selection")
    parser.add_argument("--max_jobs", type=int, default=4, help="Max number of parallel jobs to run (default: 1)")    
    parser.add_argument("--rf_n_estimators", type=int, default=100, help="RF: Number of estimators")
    parser.add_argument("--rf_max_depth", type=int, default=None, help="RF: Max depth")
    parser.add_argument("--gb_n_estimators", type=int, default=100, help="GB: Number of estimators")
    parser.add_argument("--gb_learning_rate", type=float, default=0.1, help="GB: Learning rate")
    parser.add_argument("--lr_C", type=float, default=1.0, help="LR: Inverse regularization")
    parser.add_argument("--svm_C", type=float, default=1.0, help="SVM: Regularization")    
    parser.add_argument("--svm_kernel", type=str, default='rbf', help="SVM: Kernel ('linear', 'rbf', 'poly', 'sigmoid')")
    parser.add_argument("--ab_n_estimators", type=int, default=50, help="AB: Number of estimators")
    parser.add_argument("--ab_learning_rate", type=float, default=1.0, help="AB: Learning Rate")
    args = parser.parse_args()

    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global MAX_JOBS
    MAX_JOBS = args.max_jobs
    # --- Load Data --- #
    
    data = load_data(args.blacklist, args.whitelist, args.skip_errors)

    with console.status("Engineering features...", spinner="dots"):
        data = feature_engineering(data)
        # Apply numeric conversion *after* feature engineering:
        engineered_cols = [col for col in data.columns if col not in ['fqdn', 'label']]
        data[engineered_cols] = data[engineered_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    
    data = pd.get_dummies(data, columns=['tld'], prefix='tld', dummy_na=False)
     # Apply numeric conversion *after* one-hot encoding:
    engineered_cols = [col for col in data.columns if col not in ['fqdn', 'label']]
    data[engineered_cols] = data[engineered_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- Train-Test Split ---
    train_data, test_data = train_test_split(data, test_size=args.test_size, random_state=args.random_state, stratify=data['label'])
    # Apply numeric conversion *after* splitting:
    train_engineered_cols = [col for col in train_data.columns if col not in ['fqdn', 'label']]
    train_data[train_engineered_cols] = train_data[train_engineered_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    test_engineered_cols = [col for col in test_data.columns if col not in ['fqdn', 'label']]
    test_data[test_engineered_cols] = test_data[test_engineered_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    
    from rich.panel import Panel

    # --- Initial Feature Selection (before model training)---
    initial_features = [col for col in train_data.columns if col not in ['fqdn','label', 'tld','sld','subdomain']]

    # Display training configuration summary
    summary_table = Table(title="Training Configuration", show_header=False)
    summary_table.add_row("Model", f"{args.model}")
    summary_table.add_row("Engineered Features Count", f"{len(initial_features)}")
    summary_table.add_row("Training Samples", f"{train_data.shape[0]}")
    summary_table.add_row("Test Samples", f"{test_data.shape[0]}")
    summary_table.add_row("Scaling", f"{args.scale}")
    summary_table.add_row("Quantile Transform", f"{args.quantile_transform}")
    summary_table.add_row("SMOTE", f"{args.smote}")
    summary_table.add_row("N-gram Range", f"{args.ngram_range[0]}-{args.ngram_range[1]}")
    console.print(Panel(summary_table, title="[bold cyan]Training Summary[/bold cyan]", expand=False))

    # TF-IDF Vectorizer setup (MOVED INSIDE main())
    norm = None  # Default: no regularization
    if args.l1:
        norm = 'l1'
    elif args.l2:
        norm = 'l2'

    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(args.ngram_range[0], args.ngram_range[1]), norm=norm)

    # --- Model Training/Ensembling ---
    if args.model != "ensemble":
        if args.feature_selection:
            # Iterative feature selection
            best_features, best_score, model, feature_names, best_params = iterative_feature_selection(
                train_data, tfidf_vectorizer, args.model, param_grids[args.model],
                args.scale, args.quantile_transform, args.smote, initial_features,
                num_iterations=args.num_iterations
            )
        else:
            # Train without feature selection
            model, vectorizer, feature_names, best_params = train_model(
                train_data, tfidf_vectorizer, args.model, param_grids[args.model],
                args.scale, args.quantile_transform, args.smote
            )
            console.print(f"Best parameters for {args.model}: {best_params}")
            if feature_names is None:
                feature_names = initial_features
    else:
        # Train Ensemble (VotingClassifier)
        model, vectorizer, feature_names, best_params = train_ensemble(
            train_data, tfidf_vectorizer, args.scale, args.quantile_transform,
            args.smote, param_grids
        )

    # --- Model Saving (with timestamp) ---
    if args.save_model:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{args.save_model.split('.')[0]}_{timestamp}.pkl"
        joblib.dump((model, tfidf_vectorizer, feature_names), model_filename)
        console.print(f"[green]Best model, vectorizer, and feature names saved to {model_filename}[/green]")

    # --- Prediction or Evaluation ---
    if args.predict:
        predict_fqdn(args.predict, model, tfidf_vectorizer, feature_names, args.scale, args.quantile_transform)
    else:
        evaluate_model(test_data, model, tfidf_vectorizer, feature_names, args.scale, args.quantile_transform, args.model, best_params, args.output_dir)

if __name__ == "__main__":
    main()