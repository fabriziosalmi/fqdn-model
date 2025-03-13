import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import tldextract
import joblib

# Function to extract features from FQDNs
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
    
    # Entropy calculation (measure of randomness in the domain name)
    char_count = {}
    for char in fqdn.lower():
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
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

# Load data
def load_data(whitelist_path, blacklist_path):
    # Read whitelist and blacklist files
    with open(whitelist_path, 'r') as f:
        whitelist = [line.strip() for line in f if line.strip()]
    
    with open(blacklist_path, 'r') as f:
        blacklist = [line.strip() for line in f if line.strip()]
    
    # Create dataframes
    whitelist_df = pd.DataFrame({'fqdn': whitelist, 'label': 0})  # 0 for good
    blacklist_df = pd.DataFrame({'fqdn': blacklist, 'label': 1})  # 1 for bad
    
    # Combine datasets
    df = pd.concat([whitelist_df, blacklist_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Extract features for all FQDNs
def prepare_dataset(df):
    # Extract features for each FQDN
    features_list = []
    for fqdn in df['fqdn']:
        features = extract_features(fqdn)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Combine with labels
    X = features_df
    y = df['label']
    
    return X, y

# Train the model
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 Important Features:")
    print(feature_importance.head(10))
    
    return model, X_train, X_test, y_train, y_test

# Save the model
def save_model(model, filename='fqdn_classifier_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Load the model
def load_model(filename='fqdn_classifier_model.joblib'):
    return joblib.load(filename)

# Predict new FQDNs
def predict_fqdn(model, fqdn):
    # Extract features
    features = extract_features(fqdn)
    features_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    
    result = 'Bad (Malicious)' if prediction == 1 else 'Good (Benign)'
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return result, confidence

# Main function
def main():
    # Load data
    print("Loading data...")
    df = load_data('whitelist.txt', 'blacklist.txt')
    print(f"Loaded {len(df)} FQDNs ({df['label'].value_counts()[0]} good, {df['label'].value_counts()[1]} bad)")
    
    # Prepare dataset
    print("Extracting features...")
    X, y = prepare_dataset(df)
    print(f"Extracted {X.shape[1]} features")
    
    # Train model
    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Save model
    save_model(model)
    
    # Example predictions
    print("\nExample predictions:")
    test_domains = [
        'google.com',
        'facebook.com',
        'malware-domain-example123.xyz',
        'suspicious-looking-domain.tk'
    ]
    
    for domain in test_domains:
        result, confidence = predict_fqdn(model, domain)
        print(f"{domain}: {result} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()