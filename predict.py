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
from typing import List, Tuple, Optional
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

def analyze_with_timeout(fqdn: str, default_is_bad: int = 2) -> Optional[dict]:
    """Wrapper for analyze_fqdn with timeout using signal"""
    def handler(signum, frame):
        raise TimeoutError(f"Analysis timed out after {ANALYSIS_TIMEOUT} seconds")

    # Set up the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ANALYSIS_TIMEOUT)
    
    try:
        result = analyze_fqdn(fqdn, default_is_bad)
        return result
    except TimeoutError as te:
        log.warning(f"Timeout analyzing {fqdn}: {str(te)}")
        return None
    except Exception as e:
        log.error(f"Error analyzing {fqdn}: {str(e)}")
        return None
    finally:
        signal.alarm(0)  # Disable the alarm

def load_models(model_dir):
    """Load the saved model and preprocessing components."""
    try:
        # Find model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_best_model.joblib')]
        if not model_files:
            raise FileNotFoundError("No model file found")
        model_path = os.path.join(model_dir, model_files[0])
        model = joblib.load(model_path)
        
        # Load optional preprocessing components
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
    """Make predictions for a list of FQDNs using saved model with improved error handling."""
    
    # Validate input
    if not fqdns:
        log.error("No FQDNs provided")
        return None, None
        
    if len(fqdns) > MAX_BATCH_SIZE:
        log.warning(f"Large batch detected ({len(fqdns)} FQDNs). Limiting to {MAX_BATCH_SIZE}")
        fqdns = fqdns[:MAX_BATCH_SIZE]

    # Load models and preprocessors
    model, scaler, transformer, imputer = load_models(model_dir)
    if not model:
        return None, None
        
    # Get expected features
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
        log.info(f"Using {len(expected_features)} features from model")
    else:
        expected_features = [
            'Has_A_Record', 'Has_AAAA_Record', 'Has_MX_Record', 'Has_TXT_Record',
            'Final_Protocol_HTTPS', 'Status_Code_OK', 'HTTP_to_HTTPS_Redirect',
            'High_Redirects', 'HSTS', 'Certificate_Valid', 'Has_Suspicious_Keywords',
            'Final_URL_Known', 'Certificate_Issuer', 'Certificate_Expiry', 
            'Domain_Age', 'TLD', 'Content_Type'
        ]
        log.info("Using default feature list")

    results = []
    failed_fqdns = []
    
    # Process FQDNs with retry logic
    with console.status("[bold green]Analyzing FQDNs...") as status:
        for fqdn in fqdns:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    features = analyze_with_timeout(fqdn)
                    if features:
                        # Remove non-feature columns
                        for key in ['Is_Bad', 'Overall_Status', 'IP_Analysis', 
                                  'A_Records', 'AAAA_Records', 'MX_Records', 
                                  'TXT_Records', 'Redirects']:
                            features.pop(key, None)
                        results.append(features)
                        break
                    retries += 1
                except Exception as e:
                    log.error(f"Error processing {fqdn}: {str(e)}")
                    retries += 1
                    if retries < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        
            if retries == MAX_RETRIES:
                failed_fqdns.append(fqdn)
                
    if failed_fqdns:
        log.warning(f"Failed to analyze {len(failed_fqdns)} FQDNs: {', '.join(failed_fqdns)}")
                
    if not results:
        log.error("No valid results to process")
        return None, None
        
    # Process results
    try:
        df = pd.DataFrame(results)
        
        # Handle missing features
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 2
                
        # Ensure correct feature order
        X = df[expected_features]
        
        # Apply preprocessing
        if imputer:
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns) 
        if transformer:
            X = pd.DataFrame(transformer.transform(X), columns=X.columns)
            
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        
        # Display results
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("FQDN", style="cyan")
        results_table.add_column("Prediction", style="green")
        results_table.add_column("Confidence", style="yellow")
        
        for i, fqdn in enumerate(fqdns):
            if fqdn not in failed_fqdns:
                pred = "Malicious" if predictions[i] == 1 else "Benign"
                conf = f"{max(probabilities[i])*100:.2f}%" if probabilities is not None else "N/A"
                results_table.add_row(fqdn, pred, conf)
        
        console.print("\n[bold]Prediction Results:[/]")
        console.print(results_table)
        
        return predictions, probabilities
        
    except Exception as e:
        log.error(f"Error during prediction: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Predict FQDNs using saved model")
    parser.add_argument("fqdns", nargs="+", help="One or more FQDNs to analyze")
    parser.add_argument("--model_dir", default="models", help="Directory containing saved model files")
    
    args = parser.parse_args()
    
    predict_fqdns(args.fqdns, args.model_dir)

if __name__ == "__main__":
    main()