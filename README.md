# FQDN (Fully Qualified Domain Name) Classifier

[![GitHub Issues](https://img.shields.io/github/issues/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Machine Learning classifier for predicting whether an FQDN (Fully Qualified Domain Name) is **benign** or **malicious**.

The project trains a classifier on features extracted from domain analysis (DNS, SSL, HTTP behavior, WHOIS, and lexical properties) and exposes a Flask API for inference.

> **Data Source**: The model can be trained using domain lists from [fabriziosalmi/blacklist](https://github.com/fabriziosalmi/blacklist) or any compatible blacklist/whitelist files.

## Key Features

*   **Feature Extraction**: Extracts ~22 binary/ternary features per domain including DNS record presence, SSL certificate validity, HTTP redirect behavior, HSTS, suspicious keywords, TLD risk, domain length, hyphen/digit counts, and optional WHOIS data.
*   **Multiple Model Types**: Supports Gaussian Naive Bayes, Logistic Regression, and Random Forest (default).
*   **Flask API**: REST API with input validation and health check endpoint.
*   **CLI Prediction**: Classify domains directly from the terminal using `predict.py`.
*   **Centralized Configuration**: All settings managed via `settings.py` with `config.ini` override support.
*   **Multithreaded Extraction**: Concurrent domain analysis using a configurable thread pool.

## Architecture

The project consists of three main components:

1.  **`augment.py`**: The ETL pipeline. Enriches raw FQDN lists with analysis features (DNS, HTTP, SSL, WHOIS) and writes a JSON dataset.
2.  **`fqdn_classifier.py`**: The training script. Trains a model on the augmented dataset and serializes it using `joblib`.
3.  **`api.py` / `predict.py`**: The inference layer. Loads the serialized model to serve predictions via CLI or HTTP API.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fabriziosalmi/fqdn-model.git
    cd fqdn-model
    ```

2.  **Set up environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

### 1. Training (Optional)
If you want to retrain the model with your own data or fresh data from [fabriziosalmi/blacklist](https://github.com/fabriziosalmi/blacklist):

```bash
# 1. Download lists
wget https://raw.githubusercontent.com/fabriziosalmi/blacklist/master/blacklist.txt
wget https://raw.githubusercontent.com/fabriziosalmi/blacklist/master/whitelist.txt

# 2. Augment data (extract features)
python augment.py -i blacklist.txt -o blacklist.json --is_bad Yes
python augment.py -i whitelist.txt -o whitelist.json --is_bad No

# 3. Merge & Train
python merge_datasets.py blacklist.json whitelist.json -o dataset.json
python fqdn_classifier.py dataset.json
```

### 2. Prediction (CLI)
Classify domains directly from your terminal:

```bash
python predict.py google.com
python predict.py malicious-test-domain.xyz
```

### 3. API Serving
Start the server:

```bash
python api.py
```

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"fqdn": "example.com"}'
```

---

## Configuration

The project uses a hierarchical configuration system:
1.  **Defaults**: Defined in `settings.py`.
2.  **Config File**: Values in `config.ini` override defaults.
3.  **CLI arguments**: Runtime arguments override everything.

See `settings.py` for all available options.

## Testing

Run the test suite:

```bash
pytest tests/
```

## Data Format

The training data consists of two text files:

*   **`whitelist.txt`:** Contains a list of benign FQDNs, one per line.
*   **`blacklist.txt`:** Contains a list of malicious FQDNs, one per line.

Each line should contain only the FQDN itself, without extra characters or whitespace.

**Example:**

**`whitelist.txt`:**

```
google.com
facebook.com
wikipedia.org
```

**`blacklist.txt`:**

```
malware-domain.xyz
phishing-site.tk
evil-domain.com
```

## Model Details

*   **Supported Models:** Gaussian Naive Bayes (`gaussian_nb`), Logistic Regression (`logistic_regression`), Random Forest (`random_forest`, default)
*   **Default Estimators (Random Forest):** 100 (configurable via `--rf_n_estimators`)

### Features Extracted

Features are extracted by the `analyze_fqdn` function in `augment.py`. Each feature is encoded as a binary or ternary value (0 = benign indicator, 1 = malicious indicator, 2 = unknown/unavailable):

*   DNS record presence (A, AAAA, MX, TXT, CNAME)
*   SSL certificate validity
*   HTTP status code
*   Final protocol (HTTP vs HTTPS)
*   HTTP-to-HTTPS redirect
*   Excessive redirects
*   HSTS header presence
*   Suspicious keyword detection in page content
*   SSL verification failure
*   Risky TLD
*   Domain length
*   Hyphen count
*   Digit count
*   IP address embedded in domain
*   URL shortener detection
*   Subdomain count
*   Page title and body presence
*   WHOIS data (creation date, expiration date, age) — optional, requires `--whois` flag

### Model Persistence

Trained models are saved using `joblib` in the `models/` directory and loaded automatically by `predict.py` and `api.py`.

## Performance Metrics

The `fqdn_classifier.py` script evaluates the trained model using:

*   **Accuracy:** Overall correctness of the model.
*   **ROC AUC:** Area under the receiver operating characteristic curve.
*   **Precision:** Proportion of true positives among predicted positives.
*   **Recall:** Proportion of true positives among actual positives.
*   **F1-Score:** Harmonic mean of precision and recall.
*   **Log Loss / Brier Score:** Probability calibration metrics.
*   **Confusion Matrix:** Counts of true/false positives and negatives.
*   **Feature Importance:** Ranking of features by contribution (for Random Forest and Logistic Regression).

## Contributing

Contributions are welcome! Here's how you can contribute:

1.  Fork the repository.
2.  Create a new branch: `git checkout -b feature/my-new-feature` or `git checkout -b fix/my-bug-fix`
3.  Make your changes and commit them: `git commit -am 'Add some feature'`
4.  Push to the branch: `git push origin feature/my-new-feature`
5.  Create a new Pull Request.

**Guidelines:**

*   Follow the existing code style.
*   Write clear and concise commit messages.
*   Provide tests for your changes.
*   Explain the purpose of your changes in the Pull Request description.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

*   `scikit-learn`: [https://scikit-learn.org/](https://scikit-learn.org/)
*   `tldextract`: [https://github.com/john-kurkowski/tldextract](https://github.com/john-kurkowski/tldextract)
*   `joblib`: [https://joblib.readthedocs.io/en/latest/](https://joblib.readthedocs.io/en/latest/)
*   `rich`: [https://github.com/Textualize/rich](https://github.com/Textualize/rich)
*   `Flask`: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

## Contact

If you have any questions or suggestions, feel free to open an issue.
