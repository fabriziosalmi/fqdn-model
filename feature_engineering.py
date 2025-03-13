import numpy as np
import tldextract

def extract_features(fqdn):
    # Extract domain parts using tldextract
    ext = tldextract.extract(fqdn)
    subdomain = ext.subdomain
    domain = ext.domain
    suffix = ext.suffix

    # Basic features
    features = {
        'fqdn_length': len(fqdn),
        'domain_length': len(domain),
        'subdomain_length': len(subdomain),
        'suffix_length': len(suffix),
        'num_dots': fqdn.count('.'),
        'num_hyphens': fqdn.count('-'),
        'num_underscores': fqdn.count('_'),
        'num_digits': sum(c.isdigit() for c in fqdn),
        'num_subdomains': len(subdomain.split('.')) if subdomain else 0,
        'has_www': 1 if 'www' in fqdn else 0,
        'has_subdomain': 1 if subdomain else 0,
    }
    
    # Character distribution features
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789-_.':
        features[f'char_{char}'] = fqdn.lower().count(char) / len(fqdn) if len(fqdn) > 0 else 0

    # Entropy calculation
    char_count = {}
    for char in fqdn.lower():
        char_count[char] = char_count.get(char, 0) + 1
    entropy = 0
    for count in char_count.values():
        prob = count / len(fqdn)
        entropy -= prob * np.log2(prob)
    features['entropy'] = entropy

    # Additional features
    features['consonant_ratio'] = sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' for c in fqdn) / len(fqdn) if len(fqdn) > 0 else 0
    features['vowel_ratio'] = sum(c.lower() in 'aeiou' for c in fqdn) / len(fqdn) if len(fqdn) > 0 else 0
    features['digit_ratio'] = features['num_digits'] / len(fqdn) if len(fqdn) > 0 else 0
    
    return features
