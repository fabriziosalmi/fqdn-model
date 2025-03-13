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
from rich.style import Style  # Import Style
from rich import print as rprint

from feature_engineering import extract_features

# Initialize Rich console
console = Console()

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

    # Define styles for better readability
    bold_style = Style(bold=True)
    domain_style = Style(color='yellow')
    confidence_style = Style(color=result_color)
    grey_style = Style(color="grey39")

    result_text = Text()
    result_text.append('\n🔍 FQDN Analysis Result\n\n', style=Style(bold=True, color='cyan'))
    result_text.append(f'Domain: ', style=bold_style)
    result_text.append(f'{fqdn}\n\n', style=domain_style)
    result_text.append(f'Classification: ', style=bold_style)
    result_text.append(f'{result}\n', style=Style(color=result_color)) # Directly apply color here
    result_text.append(f'Confidence: ', style=bold_style)


    # Create colored confidence bar using Spans
    bar_width = 30
    filled_chars = int(confidence * bar_width)
    empty_chars = bar_width - filled_chars
    
    # Add the filled part as a Span
    result_text.append("│" * filled_chars, style=confidence_style)

    # Add the empty part as a Span
    if empty_chars > 0:
        result_text.append("─" * empty_chars, style=grey_style) # use ─ instead of - for better appearance.

    result_text.append(f" {confidence:.2%}\n")

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