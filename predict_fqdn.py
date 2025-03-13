#!/usr/bin/env python3
import sys
import joblib
import pandas as pd
import numpy as np
import tldextract
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Initialize Rich console
console = Console()

# Function to extract features from FQDNs (copied from main script)
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

def display_prediction_result(fqdn, result, confidence):
    # Create a styled panel for the result
    is_malicious = result == 'Bad (Malicious)'
    result_color = 'red' if is_malicious else 'green'
    result_text = Text()
    result_text.append('\n🔍 FQDN Analysis Result\n\n', style='bold cyan')
    result_text.append(f'Domain: ', style='bold')
    result_text.append(f'{fqdn}\n\n', style='yellow')
    result_text.append(f'Classification: ', style='bold')
    result_text.append(f'{result}\n', style=result_color)
    result_text.append(f'Confidence: ', style='bold')
    
    # Create confidence bar
    bar_width = 30
    filled_chars = int(confidence * bar_width)
    empty_chars = bar_width - filled_chars
    confidence_bar = f'[{result_color}]{"|" * filled_chars}[/{result_color}]["+"]{"-" * empty_chars}[/] {confidence:.2%}\n'
    result_text.append(confidence_bar)
    
    # Display the panel
    console.print(Panel(result_text, border_style=result_color))

def main():
    # Check if FQDN is provided as command-line argument
    if len(sys.argv) != 2:
        console.print('\n[bold red]Error:[/] Missing domain name argument!')
        console.print('\n[bold]Usage:[/] python predict_fqdn.py <domain_name>')
        console.print('[bold]Example:[/] python predict_fqdn.py example.com\n')
        sys.exit(1)
    
    fqdn = sys.argv[1]
    
    try:
        # Show loading progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading model...", total=None)
            model = joblib.load('fqdn_classifier_model.joblib')
            
            # Make prediction
            result, confidence = predict_fqdn(model, fqdn)
            
            # Display result with rich formatting
            display_prediction_result(fqdn, result, confidence)
            
    except FileNotFoundError:
        console.print('\n[bold red]Error:[/] Model file \'fqdn_classifier_model.joblib\' not found.')
        console.print('[yellow]Please run the main script (fqdn_classifier.py) first to train and save the model.[/]\n')
        sys.exit(1)
    except Exception as e:
        console.print(f'\n[bold red]Error:[/] {str(e)}\n')
        sys.exit(1)

if __name__ == "__main__":
    main()