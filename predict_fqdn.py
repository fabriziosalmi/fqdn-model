#!/usr/bin/env python3
import sys
import time  # ...new import...
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
import argparse  # Import argparse

from feature_engineering import extract_features

# Initialize Rich console
console = Console()

# Predict new FQDNs
def predict_fqdn(model, fqdn):
    # Extract features
    features = extract_features(fqdn)
    features_df = pd.DataFrame([features])  # Or, if extract_features returns array
    # Make prediction
    prediction = model.predict(features_df)[0]  # adapt this accordingly
    probability = model.predict_proba(features_df)[0]
    
    result = 'Bad (Malicious)' if prediction == 1 else 'Good (Benign)'
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return result, confidence

def display_prediction_result(fqdn, result, confidence, exec_time):  # added exec_time parameter
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
    # Append execution time stat
    result_text.append(f'Execution Time: {exec_time:.2f} s\n', style=bold_style)

    # Display the panel
    console.print(Panel(result_text, border_style=result_color))

def main():
    parser = argparse.ArgumentParser(description="Predict FQDN maliciousness.")
    parser.add_argument("domain", nargs='?', help="The domain to check (e.g., google.com).  Omit if using --file.")
    parser.add_argument("--file", help="Path to a file containing a list of FQDNs to check, one per line.")
    parser.add_argument("--model", default='fqdn_classifier_model.joblib', help="Path to the joblib model file. Defaults to 'fqdn_classifier_model.joblib'.")

    args = parser.parse_args()

    if not args.domain and not args.file:
        console.print("[bold red]Error:[/] Must provide either a domain or a file.")
        parser.print_help()  # Show usage
        sys.exit(1)

    if args.domain and args.file:
        console.print("[bold red]Error:[/]  Cannot provide both a domain and a file.")
        parser.print_help()  # Show usage
        sys.exit(1)


    # Load model with progress indication
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading model...", total=None)
            model = joblib.load(args.model)
    except FileNotFoundError:
        console.print(f'\n[bold red]Error:[/] Model file \'{args.model}\' not found.')
        console.print('[yellow]Please run the main script (fqdn_classifier.py) first to train and save the model, or specify the correct path with --model.[/]\n')
        sys.exit(1)
    except Exception as e:
        console.print(f'\n[bold red]Error:[/] {str(e)}\n')
        sys.exit(1)

    if args.file:
        try:
            with open(args.file, 'r') as f:
                fqdns = [line.strip() for line in f if line.strip()]
        except Exception as e:
            console.print(f'\n[bold red]Error:[/] Unable to read file: {str(e)}\n')
            sys.exit(1)

        # Loop through all FQDNs in file
        for fqdn in fqdns:
            start_time = time.time()
            result, confidence = predict_fqdn(model, fqdn)
            exec_time = time.time() - start_time
            display_prediction_result(fqdn, result, confidence, exec_time)
    else:
        fqdn = args.domain
        start_time = time.time()
        result, confidence = predict_fqdn(model, fqdn)
        exec_time = time.time() - start_time
        display_prediction_result(fqdn, result, confidence, exec_time)

if __name__ == "__main__":
    main()