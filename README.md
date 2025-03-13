# fqdn-model: FQDN (Fully Qualified Domain Name) Classifier

[![GitHub Issues](https://img.shields.io/github/issues/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python-based FQDN (Fully Qualified Domain Name) classifier that uses machine learning to predict whether a given domain is benign (good) or malicious (bad).  It's built with `scikit-learn`, `tldextract`, and enhanced with `rich` for visually appealing output.

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Installation](#installation)
4.  [Usage](#usage)
    *   [Training the Model](#training-the-model)
    *   [Predicting FQDNs](#predicting-fqdns)
5.  [Data Format](#data-format)
6.  [Model Details](#model-details)
7.  [Performance Metrics](#performance-metrics)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Credits](#credits)
11. [Contact](#contact)

## Overview

This project aims to provide a reliable and easy-to-use tool for classifying FQDNs.  It leverages a Random Forest Classifier, trained on lists of known benign and malicious domains, to identify potentially harmful domains.  The use of `rich` library significantly improves the readability of the output.

## Features

*   **Classification:**  Predicts whether an FQDN is benign (good) or malicious (bad).
*   **Feature Extraction:** Extracts a comprehensive set of features from FQDNs, including length-based, character distribution, and entropy-based features.
*   **Model Training:** Trains a Random Forest Classifier on provided data.
*   **Model Persistence:** Saves and loads trained models using `joblib`.
*   **Command-Line Interface:**  A separate script (`predict_fqdn.py`) allows you to classify single FQDNs directly from the command line.
*   **Rich Output:**  Uses the `rich` library to provide visually appealing and informative output, including:
    *   Accuracy and ROC AUC scores.
    *   Confusion matrix.
    *   Classification report.
    *   Feature importance ranking.
    *   Styled prediction results.
*   **Clear Error Handling:** Provides informative error messages for common issues like missing model files or incorrect usage.
*   **Progress Bar:** Uses Rich progress bar for loading model

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

### Predicting FQDNs

1.  **Make sure the model is trained and saved:**  Run the training script (`fqdn_classifier.py`) first.

2.  **Use the prediction script (`predict_fqdn.py`)**

    ```bash
    python predict_fqdn.py <domain_name>
    ```

    Replace `<domain_name>` with the FQDN you want to classify.

    **Example:**

    ```bash
    python predict_fqdn.py google.com
    python predict_fqdn.py malware.example.xyz
    ```

    The script will load the trained model, extract features from the input FQDN, predict its class (benign or malicious), and display the result with a confidence score.

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
*   **ROC AUC:**  Area Under the Receiver Operating Characteristic curve; a measure of the model's ability to distinguish between classes.
*   **Precision:** The proportion of correctly identified malicious domains out of all domains predicted as malicious.
*   **Recall:** The proportion of correctly identified malicious domains out of all actual malicious domains.
*   **F1-Score:** The harmonic mean of precision and recall.
*   **Confusion Matrix:**  A table showing the counts of true positives, true negatives, false positives, and false negatives.
*   **Feature Importance:**  A ranking of the features based on their contribution to the model's performance.

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

## Contact

Fabrizio Salmi - [https://github.com/fabriziosalmi](https://github.com/fabriziosalmi)

If you have any questions or suggestions, feel free to open an issue or contact me directly.
