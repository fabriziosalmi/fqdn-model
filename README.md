# FQDN (Fully Qualified Domain Name) Classifier

[![GitHub Issues](https://img.shields.io/github/issues/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python-based FQDN (Fully Qualified Domain Name) classifier that uses machine learning to predict whether a given domain is benign (good) or malicious (bad). It's built with `scikit-learn`, `tldextract`, enhanced with `rich` for visually appealing output, and includes a Flask API for easy integration.

## Demo

![screenshot](https://github.com/fabriziosalmi/fqdn-model/blob/main/screenshot.png?raw=true)

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Installation](#installation)
4.  [Usage](#usage)
    *   [Training the Model](#training-the-model)
    *   [Predicting FQDNs via Command Line](#predicting-fqdns-via-command-line)
    *   [Using the Flask API](#using-the-flask-api)
5.  [Data Format](#data-format)
6.  [Model Details](#model-details)
7.  [Performance Metrics](#performance-metrics)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Credits](#credits)
11. [Contact](#contact)

## Overview

This project provides a reliable and easy-to-use tool for classifying FQDNs.  It leverages a Random Forest Classifier, trained on lists of known benign and malicious domains, to identify potentially harmful domains. The inclusion of a Flask API enables seamless integration into other applications. The `rich` library provides visual enhancements to the output.

## Features

*   **Classification:** Predicts whether an FQDN is benign (good) or malicious (bad).
*   **Feature Extraction:** Extracts a comprehensive set of features from FQDNs, including length-based, character distribution, and entropy-based features.
*   **Model Training:** Trains a Random Forest Classifier on provided data.
*   **Model Persistence:** Saves and loads trained models using `joblib`.
*   **Command-Line Interface:** Classify single FQDNs or lists of FQDNs from a file using `predict_fqdn.py`.
*   **Flask API:** Provides a RESTful API for making predictions programmatically.
*   **Rich Output:** Uses the `rich` library to provide visually appealing and informative output, including:
    *   Accuracy and ROC AUC scores.
    *   Confusion matrix.
    *   Classification report.
    *   Feature importance ranking.
    *   Styled prediction results with execution time.
*   **Clear Error Handling:** Provides informative error messages for common issues like missing model files or incorrect usage.
*   **Progress Bar:** Uses Rich progress bar for loading model
*   **Execution Time Measurement:**  Measures and displays the execution time for predictions.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/fabriziosalmi/fqdn-model.git
    cd fqdn-model
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **requirements.txt:**

    ```
    numpy
    pandas
    scikit-learn
    tldextract
    joblib
    rich
    Flask
    ```

## Usage

### Training the Model

1.  **Prepare your data:**

    *   Create two text files: `whitelist.txt` (containing a list of benign domains, one per line) and `blacklist.txt` (containing a list of malicious domains, one per line).  Make sure each line contains only a domain name without any leading or trailing whitespace.

2.  **Run the training script:**

    ```bash
    python fqdn_classifier.py
    ```

    This will:

    *   Load the data from `whitelist.txt` and `blacklist.txt`.
    *   Extract features from the FQDNs.
    *   Train a Random Forest Classifier.
    *   Evaluate the model and print performance metrics.
    *   Save the trained model to `fqdn_classifier_model.joblib`.

    The output will be visually enhanced with `rich` to provide clear and informative results.

### Predicting FQDNs via Command Line

1.  **Make sure the model is trained and saved:** Run the training script (`fqdn_classifier.py`) first.

2.  **Use the prediction script (`predict_fqdn.py`):**

    You can predict a single FQDN:

    ```bash
    python predict_fqdn.py <domain_name>
    ```

    Or, you can predict multiple FQDNs from a file (one FQDN per line):

    ```bash
    python predict_fqdn.py --file <file_path>
    ```

    Replace `<domain_name>` with the FQDN you want to classify and `<file_path>` with the path to your file.

    **Examples:**

    ```bash
    python predict_fqdn.py google.com
    python predict_fqdn.py --file domains_to_check.txt
    ```

    The script will load the trained model, extract features from the input FQDN(s), predict the class (benign or malicious), and display the result with a confidence score and execution time.

### Using the Flask API

1.  **Make sure the model is trained and saved:**  Run the training script (`fqdn_classifier.py`) first.

2.  **Run the Flask API:**

    ```bash
    python api.py
    ```

    This will start a Flask development server.  By default, it will run on `http://127.0.0.1:5000/`.

3.  **Make predictions using a POST request:**

    You can use tools like `curl`, `httpie`, or Python's `requests` library to send a POST request to the `/predict` endpoint.

    **Example using `curl`:**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"fqdn": "example.com"}' http://127.0.0.1:5000/predict
    ```

    **Example using Python's `requests` library:**

    ```python
    import requests
    import json

    url = 'http://127.0.0.1:5000/predict'
    data = {'fqdn': 'malware.example.xyz'}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

    The API will return a JSON response with the classification result, confidence, and execution time:

    ```json
    {
        "fqdn": "malware.example.xyz",
        "classification": "Bad (Malicious)",
        "confidence": "95.23%",
        "execution_time": "0.01 s"
    }
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
*   **Feature Extraction:** Features are extracted using the `extract_features` function in `feature_engineering.py`.  These features include:
    *   Length of the FQDN, domain, subdomain, and suffix.
    *   Number of dots, hyphens, and underscores.
    *   Number of digits.
    *   Number of subdomains.
    *   Presence of "www".
    *   Character distribution.
    *   Entropy.
    *   Consonant, vowel, and digit ratios.

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

Fabrizio Salmi - [https://github.com/fabriziosalmi](https://github.com/fabriziosalmi)

If you have any questions or suggestions, feel free to open an issue or contact me directly.
