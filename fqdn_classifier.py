import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import re
import tldextract
import joblib

from feature_engineering import extract_features

# Added rich for improved visualization
from rich.console import Console
from rich.table import Table

console = Console()

# Load data
def load_data(whitelist_path, blacklist_path):
    # Read whitelist and blacklist files
    try:
        with open(whitelist_path, 'r') as f:
            whitelist = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Whitelist file not found at {whitelist_path}")
        return None

    try:
        with open(blacklist_path, 'r') as f:
            blacklist = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Blacklist file not found at {blacklist_path}")
        return None

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
    # Convert to float16 *BEFORE* training if possible, otherwise leave as is
    try:
        X = X.astype(np.float16)
        #y = y.astype(np.float16) # Consider whether the labels need to be float16
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/] Could not convert features to float16: {e}. Training with default precision.")

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
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    console.print(f"Accuracy: {accuracy:.4f}", style="bold blue")
    console.print(f"ROC AUC: {roc_auc:.4f}", style="bold blue")

    # Updated confusion matrix table with consistent colors
    table = Table(title="Confusion Matrix", style="blue")
    table.add_column("", justify="center")
    table.add_column("Predicted 0", justify="center")
    table.add_column("Predicted 1", justify="center")
    table.add_row("Actual 0", str(conf_matrix[0,0]), str(conf_matrix[0,1]))
    table.add_row("Actual 1", str(conf_matrix[1,0]), str(conf_matrix[1,1]))
    console.print(table)

    console.print("Classification Report:", style="bold blue")
    console.print(report, style="blue")

    # Display additional metrics in a summary table
    metrics_table = Table(title="Metrics Summary", style="blue")
    metrics_table.add_column("Metric", justify="left")
    metrics_table.add_column("Value", justify="center")
    metrics_table.add_row("Accuracy", f"{accuracy:.4f}")
    metrics_table.add_row("ROC AUC", f"{roc_auc:.4f}")
    metrics_table.add_row("Precision", f"{precision:.4f}")
    metrics_table.add_row("Recall", f"{recall:.4f}")
    metrics_table.add_row("F1 Score", f"{f1:.4f}")
    console.print(metrics_table)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    console.print("Top 10 Important Features:", style="bold blue")
    console.print(feature_importance.head(10), style="blue")
    console.print("\n")

    return model, X_train, X_test, y_train, y_test

# Save the model
def save_model(model, filename='fqdn_classifier_model.joblib'):
    joblib.dump(model, filename)
    console.print(f"Model saved as {filename}", style="bold green")

# Load the model
def load_model(filename='fqdn_classifier_model.joblib'):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Model file not found at {filename}")
        return None
    except Exception as e:
        console.print(f"[bold red]Error:[/] Could not load model due to: {e}")
        return None

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
    console.print("Loading data...", style="bold blue")
    df = load_data('whitelist.txt', 'blacklist.txt')

    if df is None:
        console.print("[bold red]Error:[/] Could not load data. Exiting.")
        return

    console.print(f"Loaded {len(df)} FQDNs ({df['label'].value_counts()[0]} good, {df['label'].value_counts()[1]} bad)", style="green")

    console.print("Extracting features...", style="bold blue")
    X, y = prepare_dataset(df)
    console.print(f"Extracted {X.shape[1]} features", style="green")

    console.print("Training model...", style="bold blue")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    save_model(model)

    console.print("\nExample predictions:", style="bold blue")
    test_domains = [
        'google.com',
        'facebook.com',
        'malware-domain-example123.xyz',
        'suspicious-looking-domain.tk'
    ]

    for domain in test_domains:
        result, confidence = predict_fqdn(model, domain)
        console.print(f"{domain}: {result} (Confidence: {confidence:.4f})", style="bold yellow")

if __name__ == "__main__":
    main()