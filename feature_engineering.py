import numpy as np
import tldextract
from collections import Counter
import math
import re
import pandas as pd  # Import pandas

# Initialize tldextract outside the function for caching
extractor = tldextract.TLDExtract(cache_dir=".tldextract_cache")


def extract_features(fqdn: str) -> dict:
    """Extracts features from an FQDN.

    Args:
        fqdn (str): The FQDN to extract features from.

    Returns:
        dict: A dictionary of features.
    """

    ext = extractor(fqdn)
    subdomain = ext.subdomain
    domain = ext.domain
    suffix = ext.suffix

    fqdn_length = len(fqdn)
    normalized_fqdn = fqdn.lower() if fqdn else ""  # Handle empty FQDNs

    # Helper functions for readability and reuse
    def count_substring(substring: str) -> int:
        return normalized_fqdn.count(substring)

    def calculate_ratio(numerator: int, denominator: int) -> float:  # Pass denominator
        if denominator == 0:
            return 0.0  # Or some other suitable default
        return numerator / denominator

    # Basic features
    features = {
        'fqdn_length': fqdn_length,
        'domain_length': len(domain),
        'subdomain_length': len(subdomain),
        'suffix_length': len(suffix),
        'num_dots': count_substring('.'),
        'num_hyphens': count_substring('-'),
        'num_underscores': count_substring('_'),
        'num_digits': sum(c.isdigit() for c in fqdn),
        'num_subdomains': len(subdomain.split('.')) if subdomain else 0,
        'has_www': 1 if 'www' in fqdn else 0,
        'has_subdomain': 1 if subdomain else 0,
    }

    # Character distribution features
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789-_.':
        features[f'char_{char}'] = calculate_ratio(count_substring(char), fqdn_length) # Pass denominator

    # Entropy calculation
    if fqdn_length > 0:
        char_counts = Counter(normalized_fqdn)
        probabilities = [count / fqdn_length for count in char_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
    else:
        entropy = 0.0  # Explicit float

    features['entropy'] = entropy

    # Additional features
    features['consonant_ratio'] = calculate_ratio(sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' for c in fqdn), fqdn_length)  # Pass denominator
    features['vowel_ratio'] = calculate_ratio(sum(c.lower() in 'aeiou' for c in fqdn), fqdn_length)  # Pass denominator
    features['digit_ratio'] = calculate_ratio(features['num_digits'], fqdn_length) # Pass denominator

    # Added new features:
    features['letter_count'] = sum(c.isalpha() for c in fqdn)
    features['special_char_count'] = sum((not c.isalnum()) and c not in ['.', '-', '_'] for c in fqdn)

    # Additional extra features:
    features['uppercase_count'] = sum(c.isupper() for c in fqdn)
    features['unique_char_count'] = len(set(normalized_fqdn))
    features['unique_char_ratio'] = calculate_ratio(features['unique_char_count'], fqdn_length) # Pass denominator

    # --- New Features ---
    # 1.  Ratio of digits to letters
    features['digit_to_letter_ratio'] = calculate_ratio(features['num_digits'], features['letter_count'] + 1) # Pass denominator, avoid zero
    # 2.  Consecutive consonant count
    consonants = 'bcdfghjklmnpqrstvwxyz'
    max_consecutive_consonants = 0
    current_consecutive_consonants = 0
    for char in normalized_fqdn:
        if char in consonants:
            current_consecutive_consonants += 1
            max_consecutive_consonants = max(max_consecutive_consonants, current_consecutive_consonants)
        else:
            current_consecutive_consonants = 0
    features['max_consecutive_consonants'] = max_consecutive_consonants

    # 3.  Consecutive digit count
    max_consecutive_digits = 0
    current_consecutive_digits = 0
    for char in normalized_fqdn:
        if char.isdigit():
            current_consecutive_digits += 1
            max_consecutive_digits = max(max_consecutive_digits, current_consecutive_digits)
        else:
            current_consecutive_digits = 0
    features['max_consecutive_digits'] = max_consecutive_digits

    # 4.  Ratio of special characters to total length
    features['special_char_ratio'] = calculate_ratio(features['special_char_count'], fqdn_length) # Pass denominator

    # 5.  Ratio of uppercase to lowercase letters
    features['uppercase_to_lowercase_ratio'] = calculate_ratio(features['uppercase_count'], (features['letter_count'] - features['uppercase_count']) + 1) # Pass denominator, avoid zero

    # 6.  Length of longest subdomain part
    if subdomain:
        subdomain_parts = subdomain.split('.')
        features['longest_subdomain_part_length'] = max(len(part) for part in subdomain_parts)
    else:
        features['longest_subdomain_part_length'] = 0

   # 7.  Number of word-like segments (very basic) - splits on non-alphanumeric
    word_segments = re.findall(r'[a-z]+', normalized_fqdn)
    features['num_word_segments'] = len(word_segments)

    # 8.  Ratio of word segments to total length
    features['word_segment_ratio'] = calculate_ratio(features['num_word_segments'], fqdn_length) # Pass denominator

    # 9.  Presence of common keywords (e.g., "login", "bank", "secure")
    common_keywords = ['login', 'bank', 'secure', 'account', 'verify', 'update', 'paypal', 'amazon']
    for keyword in common_keywords:
        features[f'has_keyword_{keyword}'] = 1 if keyword in normalized_fqdn else 0

    # 10.  Number of repeated characters (e.g., "gooooogle.com")
    repeated_char_count = 0
    for i in range(len(normalized_fqdn) - 1):
        if normalized_fqdn[i] == normalized_fqdn[i+1]:
            repeated_char_count +=1
    features['repeated_char_count'] = repeated_char_count

    # 11. Ratio of repeated characters
    features['repeated_char_ratio'] = calculate_ratio(repeated_char_count, fqdn_length) # Pass denominator

    # 12. Whether domain and subdomain are same
    features['domain_subdomain_same'] = int(domain in subdomain)

    # 13. Whether suffix is a common TLD
    common_tlds = ['com', 'org', 'net', 'co', 'us', 'uk', 'info', 'biz']  # Expand this list
    features['is_common_tld'] = int(suffix in common_tlds)

    # 14 ratio special char to letter
    features['special_to_letter_ratio'] = calculate_ratio(features['special_char_count'], features['letter_count'] + 1) # Pass denominator, avoid zero

   # 15 Length of the domain name.
    features['len_domain'] = len(domain)

    # 16 Length of the subdomain part
    features['len_subdomain'] = len(subdomain)

   # 17 Domain part contains digits
    features['domain_has_digits'] = int(any(char.isdigit() for char in domain))

    # 18 Subdomain part contains digits
    features['subdomain_has_digits'] = int(any(char.isdigit() for char in subdomain))

   # 19 Ratio of subdomain length to domain length
    features['sub_to_domain_ratio'] = calculate_ratio(len(subdomain), (len(domain)+ 1))  # Pass denominator

    # 20 Ratio of special chars to entropy
    features['special_to_entropy_ratio'] = calculate_ratio(features['special_char_count'], features['entropy'] + 1)  # Pass denominator

    # -- New Features --
    # 21 Ratio of vowel to consonant
    features['vowel_to_consonant_ratio'] = calculate_ratio(sum(c in 'aeiou' for c in normalized_fqdn), (sum(c in 'bcdfghjklmnpqrstvwxyz' for c in normalized_fqdn)+1))

    # 22 Number of unique word segments
    features['num_unique_word_segments'] = len(set(word_segments))

    # 23 Ratio of unique word segments to total word segments
    features['unique_word_segment_ratio'] = calculate_ratio(features['num_unique_word_segments'], (features['num_word_segments']+1)) #1

    # 24 Shanon Equitability (measure of evenness in character distribution)
    if fqdn_length > 0:
        equitability = features['entropy'] / (math.log2(features['unique_char_count'] + 1) if features['unique_char_count'] > 0 else 1.0)
    else:
        equitability = 0.0
    features['shannon_equitability'] = equitability

    # 25 Whether the domain part contains a hyphen
    features['domain_has_hyphen'] = int("-" in domain)

    # 26 Whether the subdomain part contains a hyphen
    features['subdomain_has_hyphen'] = int("-" in subdomain)

    # 27 Number of consecutive vowels
    vowels = "aeiou"
    max_consecutive_vowels = 0
    current_consecutive_vowels = 0
    for char in normalized_fqdn:
        if char in vowels:
            current_consecutive_vowels += 1
            max_consecutive_vowels = max(max_consecutive_vowels, current_consecutive_vowels)
        else:
            current_consecutive_vowels = 0
    features['max_consecutive_vowels'] = max_consecutive_vowels

    # 28 Ratio of longest consecutive character sequence to total length
    def longest_consecutive_sequence(s):
      if not s:
        return 0
      max_len = 1
      curr_len = 1
      for i in range(1, len(s)):
        if s[i] == s[i-1]:
          curr_len += 1
        else:
          max_len = max(max_len, curr_len)
          curr_len = 1
      return max(max_len, curr_len)

    features['longest_consecutive_char_ratio'] = calculate_ratio(longest_consecutive_sequence(normalized_fqdn), fqdn_length) # Pass denominator

    # 29 Whether the TLD is numeric
    features['tld_is_numeric'] = int(suffix.isdigit())

    # 30 Number of distinct characters in subdomain
    features['num_distinct_chars_subdomain'] = len(set(subdomain))

    # 31 Ratio of distinct characters in the subdomain
    features['ratio_distinct_chars_subdomain'] = calculate_ratio(features['num_distinct_chars_subdomain'], len(subdomain)+1)  #2

    # 32 Number of distinct characters in domain
    features['num_distinct_chars_domain'] = len(set(domain))

    # 33 Ratio of distinct characters in the domain
    features['ratio_distinct_chars_domain'] = calculate_ratio(features['num_distinct_chars_domain'], len(domain)+1)  #3

    # 34 Jaccard Similarity between domain and subdomain character sets
    domain_chars = set(domain)
    subdomain_chars = set(subdomain)
    intersection = len(domain_chars.intersection(subdomain_chars))
    union = len(domain_chars.union(subdomain_chars))
    jaccard_similarity = calculate_ratio(intersection, (union+1))
    features['domain_subdomain_jaccard'] = jaccard_similarity

    # 35 Checks is the length of the suffix is greater than three
    features['long_suffix'] = int(len(suffix) > 3)

    # 36 Count total number of segments (split by dots)
    segments = fqdn.split('.')
    features['total_segments'] = len(segments)

    # 37 average segment length
    segment_lengths = [len(segment) for segment in segments]
    features['avg_segment_length'] = np.mean(segment_lengths) if segments else 0.0

    # 38 is short fqdn
    features['short_fqdn'] = int(fqdn_length < 10)

    # 39 Levenshtein Distance between domain and subdomain
    def levenshtein_distance(s1, s2):
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
    features['domain_subdomain_levenshtein'] = levenshtein_distance(domain, subdomain)

    # 40 ratio of levenshtein distance
    features['ratio_domain_subdomain_levenshtein'] = calculate_ratio(features['domain_subdomain_levenshtein'], fqdn_length)

    return features


from sklearn.impute import SimpleImputer

def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepares the dataset for training by extracting features from FQDNs and
    combining them with labels.

    Args:
        df (pd.DataFrame): DataFrame containing FQDNs and labels.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X)
                                         and the labels (y).
    """
    features_series = df['fqdn'].apply(extract_features)
    features_list = features_series.tolist()
    features_df = pd.DataFrame(features_list, index=df.index)  # Preserve index

    # Handle inf and NaN values
    features_df = features_df.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN

    imputer = SimpleImputer(strategy='constant', fill_value=0)  # Or use 'mean', 'median'
    X = imputer.fit_transform(features_df) # Impute NaN values with 0

    # REMOVE LESS USEFUL FEATURES
    less_useful_features = [
        'domain_has_hyphen',
        'subdomain_has_hyphen',
        'tld_is_numeric',
        'short_fqdn',
        'has_www',
        'has_subdomain'
    ]

    # Ensure the feature exists before attempting to remove
    existing_less_useful = [f for f in less_useful_features if f in features_df.columns]

    if existing_less_useful:
        X = features_df.drop(columns=existing_less_useful, errors='ignore')
        X = imputer.fit_transform(X) # Re-impute after dropping

    y = df['label']
    return X, y