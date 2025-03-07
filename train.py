"""
FQDN classifier module.
"""

# Standard Library Imports
import argparse
import datetime
import functools
import os
import re
import sys
import warnings
from collections import Counter

# Third-Party Library Imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tldextract  # Keep this separate since it's less common
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.table import Table
from rich.panel import Panel
from rich import box  # new import for consistent box style

# scikit-learn (sklearn) Imports - Grouped by Submodule
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                       train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

# imbalanced-learn (imblearn) Imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# --- Data Loading and Preprocessing ---
console = Console()  # Initialize console here

# Helper function to load TLD scores with error handling and a more robust path.
@functools.lru_cache(maxsize=1)
def load_tld_scores(file_path='tlds.csv'):
    try:
        # Use absolute path if relative path fails
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)

        df = pd.read_csv(file_path)
        # Ensure keys have a leading dot
        return {'.' + row['tld']: float(row['score']) for _, row in df.iterrows() if 'tld' in df.columns and 'score' in df.columns}
    except (FileNotFoundError, KeyError, ValueError) as e:
        console.print(f"[red]Error loading TLD scores: {e}[/red]")
        return {}

# Helper function to load word scores.  Similar improvements as load_tld_scores.
@functools.lru_cache(maxsize=1)
def load_word_scores(file_path='words.csv'):
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)

        df = pd.read_csv(file_path)
        return {row['word']: float(row['score']) for _, row in df.iterrows() if 'word' in df.columns and 'score' in df.columns}
    except (FileNotFoundError, KeyError, ValueError) as e:
        console.print(f"[red]Error loading word scores: {e}[/red]")
        return {}

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
        console.print(f"[red]Error loading CSV files: {e}[/red]")
        sys.exit(1)

    data['fqdn'] = data['fqdn'].str.lower()
    data = data[data['fqdn'].str.match(r'^[a-z0-9.-]+$')]
    if data.empty:
        console.print("[red]Error: No valid data loaded.[/red]")
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

    features['fqdn_length'] = df['fqdn'].str.len() #.apply(len)
    features['num_dots'] = df['fqdn'].str.count(r'\.')
    features['num_digits'] = df['fqdn'].str.count(r'\d')
    features['num_hyphens'] = df['fqdn'].str.count(r'-')
    features['num_non_alphanumeric'] = df['fqdn'].str.count(r'[^a-zA-Z0-9]') #
    features['entropy'] = df['fqdn'].apply(calculate_entropy)
    features['has_hyphen'] = df['fqdn'].str.contains(r'-').astype(int)
    extracted = df['fqdn'].apply(tldextract.extract)
    features['tld'] = extracted.apply(lambda x: x.suffix)
    features['sld'] = extracted.apply(lambda x: x.domain)
    features['subdomain'] = extracted.apply(lambda x: x.subdomain)
    features['tld_length'] = features['tld'].str.len()  #.apply(len)
    features['sld_length'] = features['sld'].str.len() #.apply(len)
    features['subdomain_length'] = features['subdomain'].str.len() #.apply(len)
    # [NEW] New clever feature: ratio of alphanumeric characters in subdomain to its length.
    features['subdomain_alphanum_ratio'] = features['subdomain'].apply(lambda x: sum(c.isalnum() for c in x) / (len(x) + 1e-9))
    features['digit_ratio'] = features['num_digits'] / (features['fqdn_length'] + 1e-9)
    features['non_alphanumeric_ratio'] = features['num_non_alphanumeric'] / (features['fqdn_length'] + 1e-9)
    features['starts_with_digit'] = df['fqdn'].str[0].str.isdigit().fillna(False).astype(int)
    features['ends_with_digit'] = df['fqdn'].str[-1].str.isdigit().fillna(False).astype(int)

    def vowel_consonant_ratio(text):
        vowels = "aeiou"
        vowel_count = sum(1 for char in text if char in vowels)
        consonant_count = sum(1 for char in text if char.isalpha() and char not in vowels)
        return vowel_count / (consonant_count + 1e-9)

    features['vowel_consonant_ratio'] = df['fqdn'].apply(vowel_consonant_ratio)
    for char in "._-":
        features[f'count_{char}'] = df['fqdn'].str.count(re.escape(char))

    features['longest_consecutive_digit'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x, "0123456789"))
    features['longest_consecutive_consonant'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x.lower(), "bcdfghjklmnpqrstvwxyz"))
    features['num_subdomains'] = features['subdomain'].apply(lambda x: len(x.split('.')) if x else 0)
    features['sld_entropy'] = features['sld'].apply(calculate_entropy)
    features['subdomain_entropy'] = features['subdomain'].apply(calculate_entropy)
    features['subdomain_ratio'] = features['subdomain_length'] / (features['fqdn_length'] + 1e-9)
    features['num_vowels'] = df['fqdn'].str.count(r'[aeiou]')
    features['longest_consecutive_vowel'] = df['fqdn'].apply(lambda x: longest_consecutive_chars(x.lower(), "aeiou"))
    features['unique_char_count'] = df['fqdn'].apply(lambda x: len(set(x)))
    features['has_www'] = df['fqdn'].str.contains(r'www\.').astype(int)
    features['num_consonants'] = df['fqdn'].str.count(r'[bcdfghjklmnpqrstvwxyz]')
    features['consonant_ratio'] = features['num_consonants'] / (features['fqdn_length'] + 1e-9)
    features['unique_vowels'] = df['fqdn'].apply(lambda x: len(set(char for char in x if char in "aeiou")))
    features['unique_consonants'] = df['fqdn'].apply(lambda x: len(set(char for char in x if char in "bcdfghjklmnpqrstvwxyz")))
    features['digit_letter_ratio'] = df['fqdn'].apply(lambda x: sum(c.isdigit() for c in x) / (sum(c.isalpha() for c in x) + 1e-9))
    features['most_common_char_freq'] = df['fqdn'].apply(lambda x: max(Counter(x).values()))
    features['double_letter_count'] = df['fqdn'].apply(lambda x: sum(1 for i in range(len(x)-1) if x[i]==x[i+1]))
    features['digit_sum'] = df['fqdn'].apply(lambda x: sum(int(char) for char in x if char.isdigit()))
    features['average_digit'] = df['fqdn'].apply(lambda x: (sum(int(char) for char in x if char.isdigit())) / (sum(c.isdigit() for c in x) + 1e-9))
    features['alpha_ratio'] = df['fqdn'].apply(lambda x: sum(c.isalpha() for c in x) / (len(x) + 1e-9))
    features['num_uppercase'] = df['fqdn'].str.count(r'[A-Z]') #.apply(lambda x: sum(1 for c in x if c.isupper()))
    features['uppercase_ratio'] = features['num_uppercase'] / (features['fqdn_length'] + 1e-9)
    features['num_vowel_clusters'] = df['fqdn'].str.findall(r'[aeiou]+').str.len()
    features['num_consonant_clusters'] = df['fqdn'].str.findall(r'[bcdfghjklmnpqrstvwxyz]+').str.len()
    features['has_ip'] = df['fqdn'].str.match(r'^(\d{1,3}\.){3}\d{1,3}$').astype(int)
    features['ratio_special_chars'] = features['num_non_alphanumeric'] / (features['fqdn_length'] + 1e-9)
    features['avg_token_length'] = df['fqdn'].apply(lambda x: np.mean([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['token_count'] = df['fqdn'].str.split(r'\.').str.len()
    features['dot_position_variance'] = df['fqdn'].apply(lambda x: np.var([i for i, c in enumerate(x) if c == '.']) / (len(x) + 1e-9) if '.' in x else 0)
    features['unique_alphanumeric'] = df['fqdn'].apply(lambda x: len(set(c for c in x if c.isalnum())))
    features['vowel_proportion'] = df['fqdn'].apply(lambda x: sum(c in "aeiou" for c in x.lower()) / (len(x) + 1e-9))
    features['consonant_proportion'] = df['fqdn'].apply(lambda x: sum(c.isalpha() and c.lower() not in "aeiou" for c in x) / (len(x) + 1e-9))
    features['unique_special_chars'] = df['fqdn'].apply(lambda x: len(set(c for c in x if not c.isalnum())))
    features['starts_with_www'] = df['fqdn'].str.startswith("www.").astype(int)
    features['ends_with_com'] = df['fqdn'].str.endswith(".com").astype(int)
    features['ends_with_org'] = df['fqdn'].str.endswith(".org").astype(int)
    features['ends_with_net'] = df['fqdn'].str.endswith(".net").astype(int)
    features['starts_with_letter'] = df['fqdn'].str[0].str.isalpha().fillna(False).astype(int)
    features['ends_with_letter'] = df['fqdn'].str[-1].str.isalpha().fillna(False).astype(int)
    # New feature to check for punycode in FQDN
    features['contains_punycode'] = df['fqdn'].str.contains(r'xn--').astype(int)
    features['tld_starts_with_vowel'] = features['tld'].apply(lambda x: 1 if x and x[0].lower() in "aeiou" else 0)
    features['sld_starts_with_vowel'] = features['sld'].apply(lambda x: 1 if x and x[0].lower() in "aeiou" else 0)
    features['subdomain_contains_www'] = features['subdomain'].str.contains(r'www').astype(int)
    features['std_token_length'] = df['fqdn'].apply(lambda x: np.std([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['max_token_length'] = df['fqdn'].apply(lambda x: max([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['min_token_length'] = df['fqdn'].apply(lambda x: min([len(token) for token in x.split('.')]) if x.split('.') else 0)
    features['contains_numeric_only_token'] = df['fqdn'].apply(lambda x: 1 if any(token.isdigit() for token in x.split('.')) else 0)
    features['count_numeric_tokens'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(token.isdigit() for token in tokens)) # Use .str.split for consistent tokenization
    features['count_long_tokens'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) > 7 for token in tokens))
    features['count_hyphen_tokens'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum('-' in token for token in tokens))
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
    features['single_char_token_count'] =  df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) == 1 for token in tokens))
    features['two_char_token_count'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) == 2 for token in tokens))
    features['three_char_token_count'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) == 3 for token in tokens))
    features['four_char_token_count'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) == 4 for token in tokens))
    features['five_char_token_count'] = df['fqdn'].str.split(r'\.').apply(lambda tokens: sum(len(token) == 5 for token in tokens))
    features['tld_entropy'] = features['tld'].apply(calculate_entropy)
    features['sld_to_tld_length_ratio'] = features['sld_length'] / (features['tld_length'] + 1e-9)
    features['subdomain_to_sld_length_ratio'] = features['subdomain_length'] / (features['sld_length'] + 1e-9)
    features['vowel_count_in_sld'] = features['sld'].str.count(r'[aeiou]')
    features['consonant_count_in_sld'] = features['sld'].str.count(r'[bcdfghjklmnpqrstvwxyz]')
    features['digit_count_in_sld'] = features['sld'].str.count(r'\d')
    features['special_char_count_in_sld'] = features['sld'].str.count(r'[^a-zA-Z0-9]')
    features['vowel_count_in_subdomain'] = features['subdomain'].str.count(r'[aeiou]')
    features['consonant_count_in_subdomain'] = features['subdomain'].str.count(r'[bcdfghjklmnpqrstvwxyz]')
    features['digit_count_in_subdomain'] = features['subdomain'].str.count(r'\d')
    features['special_char_count_in_subdomain'] = features['subdomain'].str.count(r'[^a-zA-Z0-9]')

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
        features[f'has_{keyword}'] = df['fqdn'].str.contains(keyword, regex=False).astype(int)  # Use str.contains and regex=False


    # Position of first/last digit/letter
    features['first_digit_position'] = df['fqdn'].apply(lambda x: x.find(next((c for c in x if c.isdigit()), '')) if any(c.isdigit() for c in x) else -1)
    features['last_digit_position'] = df['fqdn'].apply(lambda x: len(x) - 1 - x[::-1].find(next((c for c in x[::-1] if c.isdigit()), '')) if any(c.isdigit() for c in x) else -1)
    features['first_letter_position'] = df['fqdn'].apply(lambda x: x.find(next((c for c in x if c.isalpha()), '')) if any(c.isalpha() for c in x) else -1)
    features['last_letter_position'] = df['fqdn'].apply(lambda x:  len(x) -1 - x[::-1].find(next((c for c in x[::-1] if c.isalpha()), '')) if any(c.isalpha() for c in x) else -1)

    # Number of different character types
    features['num_different_char_types'] = df['fqdn'].apply(lambda x: sum([any(c.isdigit() for c in x), any(c.isalpha() for c in x), any(not c.isalnum() for c in x)]))

    # Ratio of (vowels + digits) to length
    features['vowel_digit_ratio'] = df['fqdn'].apply(lambda x: (sum(c in "aeiou" for c in x.lower()) + sum(c.isdigit() for c in x) ) / (len(x) + 1e-9))

    # Add new feature: lookup malicious score from TLDs
    tld_scores = load_tld_scores()
    features['tld_malicious_score'] = df['fqdn'].apply(
        lambda fqdn: tld_scores.get('.' + tldextract.extract(fqdn).suffix, 0)  # Use the dict.get() method
    )

    # [NEW] 10 additional features for spotting goodnesses or badnesses
    
    # Feature: has_unicode (1 if fqdn contains any non-ASCII character)
    features['has_unicode'] = df['fqdn'].apply(lambda x: 1 if any(ord(c) > 127 for c in x) else 0)
    
    # Feature: ratio_alphanumeric (ratio of alphanumeric characters to the fqdn length)
    features['ratio_alphanumeric'] = df['fqdn'].apply(lambda x: sum(c.isalnum() for c in x) / (len(x) + 1e-9))
    
    # Feature: vowel_diversity (number of unique vowels normalized by 5)
    features['vowel_diversity'] = df['fqdn'].apply(lambda x: len(set(c for c in x.lower() if c in 'aeiou')) / 5)
    
    # Feature: digit_position_variance (variance of positions where digits occur)
    features['digit_position_variance'] = df['fqdn'].apply(lambda x: np.var([i for i, c in enumerate(x) if c.isdigit()]) if any(c.isdigit() for c in x) else 0)
    
    # Feature: max_consecutive_special (max count of consecutive non-alphanumeric characters)
    features['max_consecutive_special'] = df['fqdn'].apply(lambda x: max((len(match) for match in re.findall(r'[^a-zA-Z0-9]+', x)), default=0))
    
    # Feature: median_token_length (median length of tokens split by '.')
    features['median_token_length'] = df['fqdn'].apply(lambda x: np.median([len(token) for token in x.split('.')]) if x.split('.') else 0)
    
    # Feature: token_length_entropy (entropy of token lengths)
    def token_length_entropy(x):
        tokens = [len(token) for token in x.split('.') if token]
        if not tokens:
            return 0
        counts = Counter(tokens)
        total = sum(counts.values())
        return -sum((count / total) * np.log2(count / total) for count in counts.values())
    features['token_length_entropy'] = df['fqdn'].apply(token_length_entropy)
    
    # Feature: punycode_count (number of occurrences of 'xn--')
    features['punycode_count'] = df['fqdn'].str.count(r'xn--')
    
    # Feature: bigram_entropy (entropy of character bigrams in the fqdn)
    def bigram_entropy(x):
        if len(x) < 2:
            return 0
        bigrams = [x[i:i+2] for i in range(len(x) - 1)]
        counts = Counter(bigrams)
        total = sum(counts.values())
        return -sum((count / total) * np.log2(count / total) for count in counts.values())
    features['bigram_entropy'] = df['fqdn'].apply(bigram_entropy)
    
    # Feature: is_subdomain_empty (1 if subdomain is empty; 0 otherwise)
    features['is_subdomain_empty'] = features['subdomain'].apply(lambda x: 1 if x == '' else 0)
    
    # Add new feature: lookup word malicious score from words.csv
    word_scores = load_word_scores()
    def compute_word_score(fqdn):
        words = re.findall(r'[a-z]+', fqdn.lower())
        if not words:
            return 0
        scores = [word_scores.get(word, 0) for word in words]
        return sum(scores) / len(scores)
    features['word_malicious_score'] = df['fqdn'].apply(compute_word_score)

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
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
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

    # --- Confusion Matrix and Classification Report ---
    console.print(Panel(f"{conf_matrix}", title="Confusion Matrix", box=box.ROUNDED))
    console.print(Panel(report, title="Classification Report", box=box.ROUNDED))

    # --- Model Settings ---
    settings_table = Table(show_header=False, box=box.ROUNDED)
    settings_table.add_row("Model", f"{model_name}")
    settings_table.add_row("Best Hyperparameters", f"{best_params}")
    settings_table.add_row("Scaling", f"{scale_data}")
    settings_table.add_row("Quantile Transform", f"{use_quantile_transform}")
    settings_table.add_row("Vectorizer", f"{type(vectorizer).__name__}")
    settings_table.add_row("N-gram Range", f"{vectorizer.ngram_range}")
    settings_table.add_row("Analyzer", f"{vectorizer.analyzer}")
    console.print(Panel(settings_table, title="Model Settings", box=box.ROUNDED))

    # --- Top 10 Feature Importances ---
    # Handle feature importances more robustly, checking for both pipeline and direct model cases.
    if hasattr(model, 'named_steps') and "model" in model.named_steps:
        model_step = model.named_steps["model"]
        if hasattr(model_step, 'feature_importances_'):
            feature_importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'):  # For Logistic Regression
            feature_importances = np.abs(model_step.coef_[0]) # Absolute values
        else:
            feature_importances = None
    elif hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances = np.abs(model.coef_[0])
    else:
        feature_importances = None

    if feature_importances is not None:
        try:
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False)

            table_fi = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
            table_fi.add_column("Feature")
            table_fi.add_column("Importance", justify="right")
            for _, row in fi_df.head(10).iterrows():
                table_fi.add_row(row['Feature'], f"{row['Importance']:.4f}")
            console.print(Panel(table_fi, title="Top 10 Feature Importances", box=box.ROUNDED))
        except Exception as e:
            console.print(f"[red]Could not determine feature importances: {e}[/red]")
    else:
        console.print("[yellow]Feature importances not available for this model.[/yellow]")


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

    fpr, tpr, _ = roc_curve(y_test, y_prob)  # Removed unused thresholds
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

    precision, recall, _ = precision_recall_curve(y_test, y_prob) # Removed unused thresholds
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

    if feature_importances is not None:  # Only plot if importances are available
        try:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), orient='h')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png")
            plt.close()
        except Exception as e:
            console.print(f"[red]Error creating feature importance plot: {e}[/red]")


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
          return random_search.best_estimator_, random_search.best_params_ # Return best_params

    best_rf, best_rf_params = tune_model(rf_pipeline, param_grids['random_forest'], 'RandomForest')
    best_gb, best_gb_params = tune_model(gb_pipeline, param_grids['gradient_boosting'], 'GradientBoosting')
    best_lr, best_lr_params = tune_model(lr_pipeline, param_grids['logistic_regression'], 'LogisticRegression')
    best_svm, best_svm_params = tune_model(svm_pipeline, param_grids['svm'], 'SVM')
    best_nb, best_nb_params = tune_model(nb_pipeline, param_grids['naive_bayes'], 'NaiveBayes')
    best_ab, best_ab_params = tune_model(ab_pipeline, param_grids['adaboost'], 'AdaBoost')

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
        # Attempt to get feature names from the preprocessor of the *first* estimator.
        # This assumes all estimators use the same preprocessor.  A more robust
        # approach might average feature importances across all estimators if they
        # differed, but that's significantly more complex.
        feature_names = estimators[0][1].named_steps['preprocessor'].get_feature_names_out()

        # If feature selection is present in the *first* estimator, apply it.
        if 'feature_selection' in estimators[0][1].named_steps:
             selected_features = estimators[0][1].named_steps['feature_selection'].get_support()
             feature_names = feature_names[selected_features]

    except AttributeError:
        # Fallback: Use engineered feature names if we can't get them from preprocessor.
        feature_names = X_train_engineered.columns
    except Exception as e:
      console.print(f"[red]Error during feature name extraction: {e}[/red]")
      feature_names = X_train_engineered.columns

    # Collect best parameters for each model in the ensemble
    ensemble_best_params = {
        'rf': best_rf_params,
        'gb': best_gb_params,
        'lr': best_lr_params,
        'svm': best_svm_params,
        'nb': best_nb_params,
        'ab': best_ab_params
    }


    return voting_clf, vectorizer, feature_names, ensemble_best_params


def iterative_feature_selection(train_data, vectorizer, model_name, param_grid, scale_data, use_quantile_transform, use_smote, initial_features, num_iterations=5, scoring='roc_auc'):
    """Performs iterative feature selection."""

    best_features = initial_features
    best_score = 0.0
    best_model = None
    best_best_params = None
    best_feature_names = None  # Initialize

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

        # Get feature importances (handle pipelines and non-pipeline models, including coef_ for LogisticRegression)
        if hasattr(model, 'named_steps') and "model" in model.named_steps:
            model_step = model.named_steps['model']
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
            elif hasattr(model_step, 'coef_'):
                importances = np.abs(model_step.coef_[0])  # Use absolute coefficients
            else:
                console.print("[yellow]Model does not have feature importances or coefficients.[/yellow]")
                return best_features, best_score, best_model, best_feature_names, best_best_params
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            console.print("[yellow]Model does not have feature importances or coefficients.[/yellow]")
            return best_features, best_score, best_model, best_feature_names, best_best_params

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

    # Get feature names *after* preprocessing.  Feature selection, if part of
    # the pipeline, will have already happened inside best_model.
    try:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        # Fallback to the engineered feature names if preprocessor names aren't available
        feature_names = X_train_engineered.columns
    except Exception as e:
        console.print(f"[red]Error getting feature names: {e}[/red]")
        feature_names = X_train_engineered.columns  # Fallback


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

def print_training_summary(config, features_count, features_sum_count):
    # Print the training configuration table as before...
    print(f"╭─────────── Training Summary ───────────╮")
    print(f"│                                        │")
    print(f"│ ┌────────────────────┬───────────────┐ │")
    print(f"│ │ Model              │ {config['model']} │ │")
    print(f"│ │ SMOTE              │ {config['smote']}         │ │")
    print(f"│ │ Scaling            │ {config['scaling']}         │ │")
    print(f"│ │ Quantile Transform │ {config['quantile_transform']}         │ │")
    print(f"│ │ N-gram Range       │ {config['ngram_range']}           │ │")
    print(f"│ │ Max Jobs           │ {config['max_jobs']}             │ │")
    print(f"│ │ Feature Selection  │ {config['feature_selection']}         │ │")
    print(f"│ │ Total Features     | {features_count}           │ │")
    print(f"│ └────────────────────┴───────────────┘ │")
    print(f"╰────────────────────────────────────────╯")

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

    # Create TF-IDF vectorizer before model training
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=tuple(args.ngram_range))

    # --- Initial Feature Selection (before model training)---
    initial_features = [col for col in train_data.columns if col not in ['fqdn','label', 'tld','sld','subdomain']]

    # Calculate features count and features sum count
    features_count = len(initial_features)
    features_sum_count = train_data[initial_features].sum().sum()

    # --- Calculate features count ---
    features_count = len(initial_features)

    # Print training summary with features count
    config = {
        "model": args.model,
        "smote": args.smote,
        "scaling": args.scale,
        "quantile_transform": args.quantile_transform,
        "ngram_range": f"{args.ngram_range[0]}-{args.ngram_range[1]}",
        "max_jobs": args.max_jobs,
        "feature_selection": args.feature_selection
    }

    print_training_summary(config, features_count, features_sum_count)

    # --- Model Training/Ensembling ---
    if args.model != "ensemble":
        if args.feature_selection:
            # Iterative feature selection
            best_features, best_score, best_model, best_feature_names, best_best_params = iterative_feature_selection(
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
                feature_names = initial_features  # Ensure feature_names is always defined
    else:
        # Train Ensemble (VotingClassifier)
        model, vectorizer, feature_names, best_params = train_ensemble(
            train_data, tfidf_vectorizer, args.scale, args.quantile_transform,
            args.smote, param_grids
        )

    # --- Model Saving (with timestamp) ---
    if args.save_model:
        # If the given save path is relative, join it with the project directory.
        if not os.path.isabs(args.save_model):
            save_path = os.path.join("/Users/fab/GitHub/fqdn_model", args.save_model)
        else:
            save_path = args.save_model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{save_path.split('.')[0]}_{timestamp}.pkl"
        joblib.dump((model, tfidf_vectorizer, feature_names), model_filename)
        console.print(f"[green]Best model, vectorizer, and feature names saved to {model_filename}[/green]")

    # --- Prediction or Evaluation ---
    if args.predict:
        predict_fqdn(args.predict, model, tfidf_vectorizer, feature_names, args.scale, args.quantile_transform)
    else:
        evaluate_model(test_data, model, tfidf_vectorizer, feature_names, args.scale, args.quantile_transform, args.model, best_params, args.output_dir)

if __name__ == "__main__":
    main()