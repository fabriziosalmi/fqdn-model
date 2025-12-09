# FQDN (Fully Qualified Domain Name) Classifier

[![GitHub Issues](https://img.shields.io/github/issues/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

A production-grade Machine Learning classifier for predicting whether an FQDN (Fully Qualified Domain Name) is **benign** or **malicious**. 

Built with security and scalability in mind, this project leverages a Random Forest Classifier trained on extensive datasets. It features a robust feature extraction pipeline (DNS, SSL, Whois, lexical analysis) and exposes a production-ready Flask API.

> **Data Source Attribution**: This model is designed to work with high-quality threat intelligence data. It specifically leverages the aggregated daily blacklists from [fabriziosalmi/blacklist](https://github.com/fabriziosalmi/blacklist), ensuring the model is trained on up-to-date real-world threats.

## 🚀 Key Features

*   **Advanced Feature Engineering**: Extracts over 20 distinct features including DNS records, SSL validity, lexical entropy, and specific keyword patterns.
*   **Production-Ready API**: A secure Flask-based REST API with input validation, health checks, and metrics.
*   **Robust Architecture**: 
    -   **Centralized Configuration**: All settings managed via `settings.py` (with `config.ini` override support).
    -   **Thread-Safe Timeouts**: Cross-platform (Windows/Linux/macOS) support for analysis timeouts.
    -   **Testing Suite**: Comprehensive `pytest` coverage for API and prediction logic.
*   **Performance**: Optimized extraction pipeline with caching and multithreading.
*   **Rich CLI**: Beautiful terminal output using the `rich` library.

## 🏗️ Architecture

The project consists of three main components:

1.  **`augment.py`**: The simplified ETL pipeline. It enriches raw FQDN lists (like those from [fabriziosalmi/blacklist](https://github.com/fabriziosalmi/blacklist)) with deep analysis features.
2.  **`fqdn_classifier.py`**: The training engine. Trains a Random Forest (or other) model and serializes it using `joblib`.
3.  **`api.py` / `predict.py`**: The inference layer. Loads the serialized model to serve predictions via CLI or HTTP API.

## 🛠️ Installation

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

## 🚦 Usage

### 1. Training (Optional)
If you want to retrain the model with your own data or fresh data from [fabriziosalmi/blacklist](https://github.com/fabriziosalmi/blacklist):

```bash
# 1. Download lists
wget https://raw.githubusercontent.com/fabriziosalmi/blacklist/master/blacklist.txt
wget https://raw.githubusercontent.com/fabriziosalmi/blacklist/master/whitelist.txt

# 2. Augment data (Extract features)
python augment.py -i blacklist.txt -o blacklist.json --is_bad 1
python augment.py -i whitelist.txt -o whitelist.json --is_bad 0

# 3. Merge & Train
python merge_datasets.py blacklist.json whitelist.json -o dataset.json
python fqdn_classifier.py dataset.json
```

### 2. Prediction (CLI)
Classify domains directly from your terminal:

```bash
python predict.py google.com
# Output: Benign (99.9%)

python predict.py malicious-test-domain.xyz
# Output: Malicious (95.2%)
```

### 3. API Serving
Start the secure production server:

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

## ⚙️ Configuration

The project uses a hierarchical configuration system:
1.  **Defaults**: Defined in `settings.py`.
2.  **Config File**: Values in `config.ini` override defaults.
3.  **Environment/CLI**: Runtime arguments override everything.

See `settings.py` for all available options.

## 🧪 Testing

We fervently believe in stability. Run the test suite to verify your environment:

```bash
pytest tests/
```

## Data Format

The training data consists of two text files:

*   **`whitelist.txt`:** Contains a list of benign FQDNs, one per line.
*   **`blacklist.txt`:** Contains a list of malicious FQDNs, one per line.

Each line in these files should contain only the FQDN itself, without any extra characters or whitespace.

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

*   **Model:** Random Forest Classifier
*   **Number of Estimators:** 100 (configurable in `fqdn_classifier.py`)

### Feature Engineering

*   Features are extracted using the `extract_features` function in `feature_engineering.py`. These features include:
    *   Length of the FQDN, domain, subdomain, and suffix.
    *   Number of dots, hyphens, and underscores.
    *   Number of digits.
    *   Number of subdomains.
    *   Presence of "www".
    *   Character distribution.
    *   Entropy.
    *   Consonant, vowel, and digit ratios.
    *   *Important:* The `extract_features` function *must* return the same data types as used during training (e.g., `np.float16` if training with reduced precision).

### Model Selection

*   The Random Forest Classifier was chosen for its balance of accuracy, interpretability, and robustness. Other models could be explored, but the Random Forest provides a good starting point.

### Model Persistence

*   Trained models are saved and loaded using `joblib` for efficient serialization and deserialization. This allows you to train the model once and then reuse it for prediction without retraining.

## Performance Metrics

The `fqdn_classifier.py` script evaluates the trained model using the following metrics:

*   **Accuracy:** The overall correctness of the model.
*   **ROC AUC:** Area Under the Receiver Operating Characteristic curve; a measure of the model's ability to distinguish between classes.
*   **Precision:** The proportion of correctly identified malicious domains out of all domains predicted as malicious.
*   **Recall:** The proportion of correctly identified malicious domains out of all actual malicious domains.
*   **F1-Score:** The harmonic mean of precision and recall.
*   **Confusion Matrix:** A table showing the counts of true positives, true negatives, false positives, and false negatives.
*   **Feature Importance:** A ranking of the features based on their contribution to the model's performance.

## Contributing

Contributions are welcome! Here's how you can contribute:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix:** `git checkout -b feature/my-new-feature` or `git checkout -b fix/my-bug-fix`
3.  **Make your changes and commit them:** `git commit -am 'Add some feature'`
4.  **Push to the branch:** `git push origin feature/my-new-feature`
5.  **Create a new Pull Request.**

**Guidelines:**

*   Follow the existing code style.
*   Write clear and concise commit messages.
*   Provide tests for your changes.
*   Explain the purpose of your changes in the Pull Request description.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

*   This project uses the following libraries:
    *   `scikit-learn`: [https://scikit-learn.org/](https://scikit-learn.org/)
    *   `tldextract`: [https://github.com/john-kurkowski/tldextract](https://github.com/john-kurkowski/tldextract)
    *   `joblib`: [https://joblib.readthedocs.io/en/latest/](https://joblib.readthedocs.io/en/latest/)
    *   `rich`: [https://github.com/Textualize/rich](https://github.com/Textualize/rich)
    *   `Flask`: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
    *   `requests`: [https://requests.readthedocs.io/en/latest/](https://requests.readthedocs.io/en/latest/) (For API example)

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me directly.

## Documentation Updates

* The documentation in this README has been updated to fix typos and improve clarity.
* All references to the prediction script now use `predict.py`.
* Additional details about configuration, such as the settings in config.ini, are now available in their own section below.

## Configuration Details

The configuration file (`config.ini`) in the repository allows you to customize parameters such as DNS resolvers, timeouts, and others. Refer to the comments within `config.ini` for detailed explanations of each parameter.
