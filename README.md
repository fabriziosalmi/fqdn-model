# FQDN (Fully Qualified Domain Name) Classifier

[![GitHub Issues](https://img.shields.io/github/issues/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fabriziosalmi/fqdn-model)](https://github.com/fabriziosalmi/fqdn-model/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python-based FQDN (Fully Qualified Domain Name) classifier that uses machine learning to predict whether a given domain is benign (good) or malicious (bad). It's built with `scikit-learn`, `tldextract`, enhanced with `rich` for visually appealing output, and includes a Flask API for easy integration.  This project now supports using custom-trained models via a command-line argument.

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
    *   [Feature Engineering](#feature-engineering)
    *   [Model Selection](#model-selection)
    *   [Model Persistence](#model-persistence)
7.  [Performance Metrics](#performance-metrics)
8.  [Quantization & Compression](#quantization--compression)
9.  [Contributing](#contributing)
10. [License](#license)
11. [Credits](#credits)
12. [Contact](#contact)

## Overview

This project provides a reliable and easy-to-use tool for classifying FQDNs. It leverages a Random Forest Classifier now enhanced with advanced FQDN analysis including extended DNS resolution (A, AAAA, MX, TXT, CNAME), SSL certificate verification, WHOIS lookups, and comprehensive feature extraction (domain length, subdomain count, etc.). Enhanced configuration options make it flexible and robust. The inclusion of a Flask API enables seamless integration into other applications. The `rich` library provides visual enhancements to the output. The command-line tool (`predict_fqdn.py`) now allows the user to specify which `.joblib` model file to load.

## Features

*   **Advanced Domain Analysis:** Augmented scripts now perform detailed DNS record lookups, SSL certificate checks, WHOIS queries, and keyword matching.
*   **Configurable Parameters:** Customize DNS resolvers, timeouts, worker threads, and WHOIS lookups via command-line flags or config files.
*   **Model Training & Persistence:** Trains a Random Forest Classifier with enhanced features and supports model quantization and joblib compression.
*   **Command-Line Interface & Flask API:** Updated scripts for improved error handling, progress reporting, and support for custom-trained or compressed models.
*   **Rich Output:** Uses the `rich` library to present stylized output and detailed performance metrics.

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
    *   The repository now includes two shipped files:
        - `whitelist.txt`: Contains a list of benign domains.
        - `blacklist.txt`: Contains a list of malicious domains.
    *   These files serve as the basis for training.

2.  **Build Augmented Datasets**

    The `augment.py` script now supports additional parameters for improved domain analysis. For example:
    ```bash
    python augment.py -i whitelist.txt -o whitelist.csv --is_bad 0 --dns-resolvers 1.1.1.1,8.8.8.8 --whois
    python augment.py -i blacklist.txt -o blacklist.csv --is_bad 1 --dns-resolvers 1.1.1.1,8.8.8.8 --whois
    ```

3.  **Merge Datasets**

    Combine the augmented CSV files using the `merge_datasets.py` script:
    ```bash
    python merge_datasets.py whitelist.csv blacklist.csv dataset.csv
    ```

4.  **Train and Save Best Model**

    Train the model using the merged dataset with the `fqdn_classifier.py` script:
    ```bash
    python fqdn_classifier.py dataset.csv
    ```

    This updated training process leverages the enhanced feature extraction and improved error handling.

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

    **Specifying a Custom Model:**

    Use the `--model` argument to specify the path to a custom-trained or compressed model:

    ```bash
    python predict_fqdn.py <domain_name> --model <path_to_model.joblib>
    ```

    ```bash
    python predict_fqdn.py --file <file_path> --model <path_to_model.joblib>
    ```

    Replace `<domain_name>` with the FQDN you want to classify, `<file_path>` with the path to your file, and `<path_to_model.joblib>` with the path to your model file.  If `--model` is not specified, the script defaults to using `fqdn_classifier_model.joblib`.

    **Examples:**

    ```bash
    python predict_fqdn.py google.com
    python predict_fqdn.py --file domains_to_check.txt
    python predict_fqdn.py malware.example.com --model my_custom_model.joblib
    python predict_fqdn.py --file domains_to_check.txt --model compressed_model.joblib
    ```

    Example output:

    ```bash
    python fqdn_classifier.py dataset.20k.json 

    [22:42:39] INFO     Data loaded successfully from dataset.20k.json                                                                                                      fqdn_classifier.py:91
            INFO     Cross-validation (accuracy) scores (mean ± std): 0.9991 ± 0.0006                                                                                   fqdn_classifier.py:193
    Cross-validating random_forest... (Fold 5/5)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Metric               ┃ Value           ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ Accuracy             │ 0.9990          │
    │ F1 Score             │ 0.9990          │
    │ Precision            │ 0.9990          │
    │ Recall               │ 0.9990          │
    │ ROC AUC              │ 1.0000          │
    │ Log Loss             │ 0.0552          │
    │ Brier Score          │ 0.0061          │
    │ CV Accuracy (Mean)   │ 0.9991          │
    │ CV Accuracy (Std)    │ 0.0006          │
    └──────────────────────┴─────────────────┘
    ╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │                                                                                                                                                                                           │
    │ Classification Report:                                                                                                                                                                    │
    │               precision    recall  f1-score   support                                                                                                                                     │
    │                                                                                                                                                                                           │
    │            0       1.00      1.00      1.00      2781                                                                                                                                     │
    │            1       1.00      1.00      1.00      1216                                                                                                                                     │
    │                                                                                                                                                                                           │
    │     accuracy                           1.00      3997                                                                                                                                     │
    │    macro avg       1.00      1.00      1.00      3997                                                                                                                                     │
    │ weighted avg       1.00      1.00      1.00      3997                                                                                                                                     │
    │                                                                                                                                                                                           │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

    Confusion Matrix:
    [[2777    4]
    [   0 1216]]
            INFO     Confusion matrix saved to confusion_matrix.png                                                                                                      fqdn_classifier.py:58

    Top 10 Feature Importances:
    ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┃ Rank   ┃ Feature                        ┃ Importance   ┃
    ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
    │ 1      │ Domain_Length                  │ 0.4474       │
    │ 2      │ Has_Suspicious_Keywords        │ 0.1301       │
    │ 3      │ Status_Code_OK                 │ 0.0860       │
    │ 4      │ Num_Digits                     │ 0.0673       │
    │ 5      │ Is_Risky_TLD                   │ 0.0598       │
    │ 6      │ Num_Hyphens                    │ 0.0409       │
    │ 7      │ Final_Protocol_HTTPS           │ 0.0393       │
    │ 8      │ High_Redirects                 │ 0.0386       │
    │ 9      │ A                              │ 0.0173       │
    │ 10     │ SSL_Verification_Failed        │ 0.0138       │
    └────────┴────────────────────────────────┴──────────────┘
            INFO     Best model and preprocessing steps saved to: models                                                                                                fqdn_classifier.py:294

    Total execution time: 0.84 seconds
    ```

    The script will load the trained model, extract features from the input FQDN(s), predict the class (benign or malicious), and display the result with a confidence score and execution time.

### Using the Flask API

1.  **Make sure the model is trained and saved:**  Run the training script (`fqdn_classifier.py`) first.

2.  **Run the Flask API:**

    ```bash
    python api.py
    ```

    This will start a Flask development server. By default, it will run on `http://127.0.0.1:5000/`.  The API always uses `fqdn_classifier_model.joblib`.

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

4. Web UI
   
> [!NOTE]
> You can also interact with the API by using the practical web interface at `http://127.0.0.1:5000/`

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

## Quantization & Compression

This project supports model quantization and compression to reduce the model size and improve performance:

*   **Training with Reduced Precision:** The `fqdn_classifier.py` script can be modified to train models using `np.float16` data types for the feature matrix. This reduces the memory footprint of the model and can potentially improve prediction speed.  See the comments in the `train_model` function in `fqdn_classifier.py` for details.  *Important: `extract_features` must also return `np.float16` features in this case.*
*   **`joblib` Compression:** The `joblib.dump` function can be used with the `compress` argument to compress the saved model file. This reduces the file size on disk. Use the `compress_model.py` script for this.

See `compress_model.py -h` for instructions:

```bash
python compress_model.py -h
```

Example usage:

```bash
python compress_model.py fqdn_classifier_model.joblib -c 5 -o compressed_model.joblib --overwrite
```

* It's now possible to use the `predict_fqdn.py` with compressed, custom model using `--model` argument

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
