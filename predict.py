import joblib
import pandas as pd
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import logging
import os
import argparse
from augment import analyze_fqdn
import signal
import time
from typing import List, Tuple, Optional, Union
import concurrent.futures

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)]
)
log = logging.getLogger("rich")

# Constants
ANALYSIS_TIMEOUT = 30  # seconds per FQDN
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_BATCH_SIZE = 100

class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass

def analyze_with_timeout(fqdn: str) -> Optional[dict]:
    """Wrapper for analyze_fqdn with timeout."""
    def handler(signum, frame):
        raise TimeoutError(f"Analysis timed out after {ANALYSIS_TIMEOUT} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ANALYSIS_TIMEOUT)

    try:
        result = analyze_fqdn(fqdn, default_is_bad_numeric=0, whois_enabled=False)
        return result
    except TimeoutError as te:
        log.warning(f"Timeout analyzing {fqdn}: {str(te)}")
        return None
    except Exception as e:
        log.error(f"Error analyzing {fqdn}: {str(e)}")
        return None
    finally:
        signal.alarm(0)

def load_models(model_dir):
    """Load the saved model and preprocessing components."""
    try:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_best_model.joblib')]
        if not model_files:
            raise FileNotFoundError("No model file found")
        model_path = os.path.join(model_dir, model_files[0])
        model = joblib.load(model_path)

        scaler = None
        if os.path.exists(os.path.join(model_dir, 'scaler.joblib')):
            scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

        transformer = None
        if os.path.exists(os.path.join(model_dir, 'quantile_transformer.joblib')):
            transformer = joblib.load(os.path.join(model_dir, 'quantile_transformer.joblib'))

        imputer = None
        if os.path.exists(os.path.join(model_dir, 'imputer.joblib')):
            imputer = joblib.load(os.path.join(model_dir, 'imputer.joblib'))

        return model, scaler, transformer, imputer

    except Exception as e:
        log.error(f"Error loading models: {e}")
        return None, None, None, None

def predict_fqdns(fqdns: List[str], model_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Make predictions for a list of FQDNs."""

    if not fqdns:
        log.error("No FQDNs provided")
        return None, None

    if len(fqdns) > MAX_BATCH_SIZE:
        log.warning(f"Large batch detected ({len(fqdns)} FQDNs).  Limiting to {MAX_BATCH_SIZE}")
        fqdns = fqdns[:MAX_BATCH_SIZE]

    model, scaler, transformer, imputer = load_models(model_dir)
    if not model:
        return None, None

    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
    elif hasattr(model, 'n_features_in_'):
        expected_features = [f"feature_{i}" for i in range(model.n_features_in_)]
    else:
        log.error("Could not determine expected features from the model.")
        return None, None
    log.info(f"Expecting {len(expected_features)} features: {', '.join(expected_features)}")

    results = []
    failed_fqdns = []

    with console.status("[bold green]Analyzing and extracting features from FQDNs...") as status:
        for fqdn in fqdns:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    analysis_result = analyze_with_timeout(fqdn)

                    if analysis_result:
                        features = {}
                        for feature in expected_features:
                            features[feature] = analysis_result.get(feature, 0)
                        results.append(features)
                        break
                    else:
                        retries += 1
                except Exception as e:
                    log.error(f"Error processing {fqdn}: {str(e)}")
                    retries += 1
                    if retries < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

            if retries == MAX_RETRIES:
                failed_fqdns.append(fqdn)

    if failed_fqdns:
        log.warning(f"Failed to analyze/extract features for {len(failed_fqdns)} FQDNs: {', '.join(failed_fqdns)}")

    if not results:
        log.error("No valid results to process")
        return None, None

    try:
        df = pd.DataFrame(results)

        for feature in expected_features:
            if feature not in df.columns:
                log.warning(f"Missing feature '{feature}'. Filling with 0 (benign imputation).")
                df[feature] = 0

        X = df[expected_features]

        if imputer:
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        if transformer:
            X = pd.DataFrame(transformer.transform(X), columns=X.columns)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        prediction_labels = ["Benign", "Malicious"]
        predicted_labels = [prediction_labels[p] for p in predictions]

        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("FQDN", style="cyan")
        results_table.add_column("Prediction", style="green")
        results_table.add_column("Confidence", style="yellow")

        result_index = 0
        for fqdn in fqdns:
            if fqdn in failed_fqdns:
                results_table.add_row(fqdn, "Analysis Failed", "N/A")
            else:
                pred = predicted_labels[result_index]
                conf = f"{max(probabilities[result_index])*100:.2f}%" if probabilities is not None else "N/A"
                results_table.add_row(fqdn, pred, conf)
                result_index += 1

        console.print("\n[bold]Prediction Results:[/]")
        console.print(results_table)

        return predictions, probabilities

    except Exception as e:
        log.error(f"Error during prediction: {str(e)}")
        return None, None

def get_fqdns_from_file(filepath: str) -> List[str]:
    """Reads FQDNs from a file, one per line."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        log.error(f"File not found: {filepath}")
        return []
    except Exception as e:
        log.error(f"Error reading file {filepath}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Predict FQDNs using a saved model.")
    parser.add_argument("fqdns", nargs="*", help="One or more FQDNs to analyze (space or comma-separated).")
    parser.add_argument("--fqdns_file", help="Path to a file containing FQDNs, one per line.")
    parser.add_argument("--model_dir", default="models", help="Directory containing saved model files.")
    args = parser.parse_args()

    all_fqdns = []

    # 1. FQDNs from command-line arguments (if any)
    if args.fqdns:
        # Handle both space and comma-separated FQDNs
        for arg in args.fqdns:
            all_fqdns.extend(arg.replace(",", " ").split())  # Split by comma or space

    # 2. FQDNs from file (if specified)
    if args.fqdns_file:
        file_fqdns = get_fqdns_from_file(args.fqdns_file)
        all_fqdns.extend(file_fqdns)

    # Remove duplicates and empty strings:
    all_fqdns = list(set(fqdn for fqdn in all_fqdns if fqdn))

    if not all_fqdns:
        log.error("No FQDNs provided via command line or file.")
        return

    predict_fqdns(all_fqdns, args.model_dir)

if __name__ == "__main__":
    main()